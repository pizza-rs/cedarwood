//!  Efficiently-updatable double-array trie in Rust (ported from cedar).
//!
//! Add it to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! cedarwood = "0.4"
//! ```
//!
//! then you are good to go. If you are using Rust 2015 you have to `extern crate cedarwood` to your crate root as well.
//!
//! ## Example
//!
//! ```rust
//! use cedarwood::Cedar;
//!
//! let dict = vec![
//!     "a",
//!     "ab",
//!     "abc",
//!     "アルゴリズム",
//!     "データ",
//!     "構造",
//!     "网",
//!     "网球",
//!     "网球拍",
//!     "中",
//!     "中华",
//!     "中华人民",
//!     "中华人民共和国",
//! ];
//! let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
//! let mut cedar = Cedar::new();
//! cedar.build(&key_values);
//!
//! let result: Vec<i32> = cedar.common_prefix_search("abcdefg").unwrap().iter().map(|x| x.0).collect();
//! assert_eq!(vec![0, 1, 2], result);
//!
//! let result: Vec<i32> = cedar
//!     .common_prefix_search("网球拍卖会")
//!     .unwrap()
//!     .iter()
//!     .map(|x| x.0)
//!     .collect();
//! assert_eq!(vec![6, 7, 8], result);
//!
//! let result: Vec<i32> = cedar
//!     .common_prefix_search("中华人民共和国")
//!     .unwrap()
//!     .iter()
//!     .map(|x| x.0)
//!     .collect();
//! assert_eq!(vec![9, 10, 11, 12], result);
//!
//! let result: Vec<i32> = cedar
//!     .common_prefix_search("データ構造とアルゴリズム")
//!     .unwrap()
//!     .iter()
//!     .map(|x| x.0)
//!     .collect();
//! assert_eq!(vec![4], result);
//! ```

use core::fmt;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// NInfo stores the information about the trie
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct NInfo {
    sibling: u8, // the index of right sibling, it is 0 if it doesn't have a sibling.
    child: u8,   // the index of the first child
}

/// Node contains the array of `base` and `check` as specified in the paper: "An efficient implementation of trie structures"
/// https://dl.acm.org/citation.cfm?id=146691
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct Node {
    base_: i32, // if it is a negative value, then it stores the value of previous index that is free.
    check: i32, // if it is a negative value, then it stores the value of next index that is free.
}

impl Node {
    #[inline]
    fn base(&self) -> i32 {
        #[cfg(feature = "reduced-trie")]
        return -(self.base_ + 1);
        #[cfg(not(feature = "reduced-trie"))]
        return self.base_;
    }
}

/// Block stores the linked-list pointers and the stats info for blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Block {
    prev: i32,   // previous block's index, 3 bytes width
    next: i32,   // next block's index, 3 bytes width
    num: i16,    // the number of slots that is free, the range is 0-256
    reject: i16, // a heuristic number to make the search for free space faster, it is the minimum number of iteration in each trie node it has to try before we can conclude that we can reject this block. If the number of kids for the block we are looking for is less than this number then this block is worthy of searching.
    trial: i32,  // the number of times this block has been probed by `find_places` for the free block.
    e_head: i32, // the index of the first empty elemenet in this block
}

impl Block {
    pub fn new() -> Self {
        Block {
            prev: 0,
            next: 0,
            num: 256,    // each of block has 256 free slots at the beginning
            reject: 257, // initially every block need to be fully iterated through so that we can reject it to be unusable.
            trial: 0,
            e_head: 0,
        }
    }
}

/// Blocks are marked as either of three categories, so that we can quickly decide if we can
/// allocate it for use or not.
enum BlockType {
    Open,   // The block has spaces more than 1.
    Closed, // The block is only left with one free slot
    Full,   // The block's slots are fully used.
}

/// `Cedar` holds all of the information about double array trie.
#[derive(Serialize, Deserialize, Clone)]
pub struct Cedar {
    array: Vec<Node>, // storing the `base` and `check` info from the original paper.
    n_infos: Vec<NInfo>,
    blocks: Vec<Block>,
    reject: Vec<i16>,
    blocks_head_full: i32,   // the index of the first 'Full' block, 0 means no 'Full' block
    blocks_head_closed: i32, // the index of the first 'Closed' block, 0 means no ' Closed' block
    blocks_head_open: i32,   // the index of the first 'Open' block, 0 means no 'Open' block
    capacity: usize,
    size: usize,
    ordered: bool,
    max_trial: i32, // the parameter for cedar, it could be tuned for more, but the default is 1.
}

impl fmt::Debug for Cedar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cedar(size={}, ordered={})", self.size, self.ordered)
    }
}

#[allow(dead_code)]
const CEDAR_VALUE_LIMIT: i32 = std::i32::MAX - 1;
const CEDAR_NO_VALUE: i32 = -1;

/// Iterator for `common_prefix_search`
#[derive(Clone)]
pub struct PrefixIter<'a> {
    cedar: &'a Cedar,
    key: &'a [u8],
    from: usize,
    i: usize,
}

impl<'a> Iterator for PrefixIter<'a> {
    type Item = (i32, usize);

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.key.len()))
    }

    fn next(&mut self) -> Option<Self::Item> {
        while self.i < self.key.len() {
            if let Some(value) = self.cedar.find(&self.key[self.i..=self.i], &mut self.from) {
                if value == CEDAR_NO_VALUE {
                    self.i += 1;
                    continue;
                } else {
                    let result = Some((value, self.i));
                    self.i += 1;
                    return result;
                }
            } else {
                break;
            }
        }

        None
    }
}

/// Iterator over the outgoing edges of a single trie node.
///
/// Yields `(byte, child_node)` pairs in cedar's native sibling order
/// (insertion-order within the same parent).  Skips the terminator
/// (byte 0) used internally to store the node's value, since callers
/// driving custom walks should consume that via [`Cedar::value_at`].
#[derive(Clone)]
pub struct ChildIter<'a> {
    cedar: &'a Cedar,
    node: usize,
    next_byte: u8,
}

impl<'a> Iterator for ChildIter<'a> {
    type Item = (u8, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let b = self.next_byte;
        if b == 0 {
            return None;
        }
        let base = self.cedar.array[self.node].base();
        if base < 0 {
            return None;
        }
        let child = (base ^ (b as i32)) as usize;
        // Defensive: a stale sibling link should never escape, but if
        // the slot is no longer owned by `node` we stop cleanly.
        match self.cedar.array.get(child) {
            Some(n) if n.check == self.node as i32 => {
                self.next_byte = self.cedar.n_infos[child].sibling;
                Some((b, child))
            }
            _ => None,
        }
    }
}

/// Iterator for `common_prefix_predict`
#[derive(Clone)]
pub struct PrefixPredictIter<'a> {
    cedar: &'a Cedar,
    key: &'a [u8],
    from: usize,
    p: usize,
    root: usize,
    value: Option<i32>,
}

impl<'a> PrefixPredictIter<'a> {
    fn next_until_none(&mut self) -> Option<(i32, usize)> {
        #[allow(clippy::never_loop)]
        while let Some(value) = self.value {
            let result = (value, self.p);

            let (v_, from_, p_) = self.cedar.next(self.from, self.p, self.root);
            self.from = from_;
            self.p = p_;
            self.value = v_;

            return Some(result);
        }

        None
    }
}

impl<'a> Iterator for PrefixPredictIter<'a> {
    type Item = (i32, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.from == 0 && self.p == 0 {
            // To locate the prefix's position first, if it doesn't exist then that means we
            // don't have do anything. `from` would serve as the cursor.
            if self.cedar.find(self.key, &mut self.from).is_some() {
                self.root = self.from;

                let (v_, from_, p_) = self.cedar.begin(self.from, self.p);
                self.from = from_;
                self.p = p_;
                self.value = v_;

                self.next_until_none()
            } else {
                None
            }
        } else {
            self.next_until_none()
        }
    }
}

#[allow(clippy::cast_lossless)]
impl Cedar {
    /// Initialize the Cedar for further use.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let mut array: Vec<Node> = Vec::with_capacity(256);
        let n_infos: Vec<NInfo> = (0..256).map(|_| Default::default()).collect();
        let mut blocks: Vec<Block> = vec![Block::new(); 1];
        let reject: Vec<i16> = (0..=256).map(|i| i + 1).collect();

        #[cfg(feature = "reduced-trie")]
        array.push(Node { base_: -1, check: -1 });
        #[cfg(not(feature = "reduced-trie"))]
        array.push(Node { base_: 0, check: -1 });

        for i in 1..256 {
            // make `base_` point to the previous element, and make `check` point to the next element
            array.push(Node {
                base_: -(i - 1),
                check: -(i + 1),
            })
        }

        // make them link as a cyclic doubly-linked list
        array[1].base_ = -255;
        array[255].check = -1;

        blocks[0].e_head = 1;

        Cedar {
            array,
            n_infos,
            blocks,
            reject,
            blocks_head_full: 0,
            blocks_head_closed: 0,
            blocks_head_open: 0,
            capacity: 256,
            size: 256,
            ordered: true,
            max_trial: 1,
        }
    }

    /// Build the double array trie from the given key value pairs
    #[allow(dead_code)]
    pub fn build(&mut self, key_values: &[(&str, i32)]) {
        for (key, value) in key_values {
            self.update(key, *value);
        }
    }

    /// Update the key for the value, it is public interface that works on &str
    pub fn update(&mut self, key: &str, value: i32) {
        let from = 0;
        let pos = 0;
        self.update_(key.as_bytes(), value, from, pos);
    }

    // Update the key for the value, it is internal interface that works on &[u8] and cursor.
    fn update_(&mut self, key: &[u8], value: i32, mut from: usize, mut pos: usize) -> i32 {
        if from == 0 && key.is_empty() {
            panic!("failed to insert zero-length key");
        }

        while pos < key.len() {
            #[cfg(feature = "reduced-trie")]
            {
                let val_ = self.array[from].base_;
                if val_ >= 0 && val_ != CEDAR_VALUE_LIMIT {
                    let to = self.follow(from, 0);
                    self.array[to as usize].base_ = val_;
                }
            }

            from = self.follow(from, key[pos]) as usize;
            pos += 1;
        }

        #[cfg(feature = "reduced-trie")]
        let to = if self.array[from].base_ >= 0 {
            from as i32
        } else {
            self.follow(from, 0)
        };

        #[cfg(feature = "reduced-trie")]
        {
            if self.array[to as usize].base_ == CEDAR_VALUE_LIMIT {
                self.array[to as usize].base_ = 0;
            }
        }

        #[cfg(not(feature = "reduced-trie"))]
        let to = self.follow(from, 0);

        self.array[to as usize].base_ = value;
        self.array[to as usize].base_
    }

    // To move in the trie by following the `label`, and insert the node if the node is not there,
    // it is used by the `update` to populate the trie.
    #[inline]
    fn follow(&mut self, from: usize, label: u8) -> i32 {
        let base = self.array[from].base();

        #[allow(unused_assignments)]
        let mut to = 0;

        // the node is not there
        if base < 0 || self.array[(base ^ (label as i32)) as usize].check < 0 {
            // allocate a e node
            to = self.pop_e_node(base, label, from as i32);
            let branch: i32 = to ^ (label as i32);

            // maintain the info in ninfo
            self.push_sibling(from, branch, label, base >= 0);
        } else {
            // the node is already there and the ownership is not `from`, therefore a conflict.
            to = base ^ (label as i32);
            if self.array[to as usize].check != (from as i32) {
                // call `resolve` to relocate.
                to = self.resolve(from, base, label);
            }
        }

        to
    }

    // Find key from double array trie, with `from` as the cursor to traverse the nodes.
    fn find(&self, key: &[u8], from: &mut usize) -> Option<i32> {
        #[allow(unused_assignments)]
        let mut to: usize = 0;
        let mut pos = 0;

        // recursively matching the key.
        while pos < key.len() {
            #[cfg(feature = "reduced-trie")]
            {
                if self.array[*from].base_ >= 0 {
                    break;
                }
            }

            to = (self.array[*from].base() ^ (key[pos] as i32)) as usize;
            if self.array[to as usize].check != (*from as i32) {
                return None;
            }

            *from = to;
            pos += 1;
        }

        #[cfg(feature = "reduced-trie")]
        {
            if self.array[*from].base_ >= 0 {
                if pos == key.len() {
                    return Some(self.array[*from].base_);
                } else {
                    return None;
                }
            }
        }

        // return the value of the node if `check` is correctly marked fpr the ownership, otherwise
        // it means no value is stored.
        let n = &self.array[(self.array[*from].base()) as usize];
        if n.check != (*from as i32) {
            Some(CEDAR_NO_VALUE)
        } else {
            Some(n.base_)
        }
    }

    /// Delete the key from the trie, the public interface that works on &str
    pub fn erase(&mut self, key: &str) {
        self.erase_(key.as_bytes())
    }

    // Delete the key from the trie, the internal interface that works on &[u8]
    fn erase_(&mut self, key: &[u8]) {
        let mut from = 0;

        // move the cursor to the right place and use erase__ to delete it.
        if let Some(v) = self.find(key, &mut from) {
            if v != CEDAR_NO_VALUE {
                self.erase__(from);
            }
        }
    }

    fn erase__(&mut self, mut from: usize) {
        #[cfg(feature = "reduced-trie")]
        let mut e: i32 = if self.array[from].base_ >= 0 {
            from as i32
        } else {
            self.array[from].base()
        };

        #[cfg(feature = "reduced-trie")]
        {
            from = self.array[e as usize].check as usize;
        }

        #[cfg(not(feature = "reduced-trie"))]
        let mut e = self.array[from].base();

        #[allow(unused_assignments)]
        let mut has_sibling = false;
        loop {
            let n = self.array[from].clone();
            has_sibling = self.n_infos[(n.base() ^ (self.n_infos[from].child as i32)) as usize].sibling != 0;

            // if the node has siblings, then remove `e` from the sibling.
            if has_sibling {
                self.pop_sibling(from as i32, n.base(), (n.base() ^ e) as u8);
            }

            // maintain the data structures.
            self.push_e_node(e);
            e = from as i32;

            // traverse to the parent.
            from = self.array[from].check as usize;

            // if it has sibling then this layer has more than one nodes, then we are done.
            if has_sibling {
                break;
            }
        }
    }

    /// To check if `key` is in the dictionary.
    pub fn exact_match_search(&self, key: &str) -> Option<(i32, usize, usize)> {
        let key = key.as_bytes();
        let mut from = 0;

        if let Some(value) = self.find(key, &mut from) {
            if value == CEDAR_NO_VALUE {
                return None;
            }

            Some((value, key.len(), from))
        } else {
            None
        }
    }

    /// To return an iterator to iterate through the common prefix in the dictionary with the `key` passed in.
    pub fn common_prefix_iter<'a>(&'a self, key: &'a str) -> PrefixIter<'a> {
        let key = key.as_bytes();

        PrefixIter {
            cedar: self,
            key,
            from: 0,
            i: 0,
        }
    }

    /// To return the collection of the common prefix in the dictionary with the `key` passed in.
    pub fn common_prefix_search(&self, key: &str) -> Option<Vec<(i32, usize)>> {
        self.common_prefix_iter(key).map(Some).collect()
    }

    /// To return an iterator to iterate through the list of words in the dictionary that has `key` as their prefix.
    pub fn common_prefix_predict_iter<'a>(&'a self, key: &'a str) -> PrefixPredictIter<'a> {
        let key = key.as_bytes();

        PrefixPredictIter {
            cedar: self,
            key,
            from: 0,
            p: 0,
            root: 0,
            value: None,
        }
    }

    /// To return the list of words in the dictionary that has `key` as their prefix.
    pub fn common_prefix_predict(&self, key: &str) -> Option<Vec<(i32, usize)>> {
        self.common_prefix_predict_iter(key).map(Some).collect()
    }

    // -----------------------------------------------------------------
    // Low-level traversal primitives
    //
    // These expose the double-array trie structure to callers that
    // implement custom walks (e.g. Levenshtein automaton intersection,
    // wildcard / regex DFA walks, n-gram pre-filters).  They are kept
    // small and orthogonal so that any caller can drive the trie one
    // edge at a time without paying for full prefix scans.
    // -----------------------------------------------------------------

    /// Index of the trie root.  All custom walks start here.
    #[inline]
    pub fn root(&self) -> usize {
        0
    }

    /// Look up the value stored at `node`, if any.
    ///
    /// In the standard (non-reduced) double-array, a node has a value
    /// when its `base ^ 0` child slot is owned by it; that slot's
    /// `base_` field holds the value.  Returns `None` for branch-only
    /// nodes and for nodes whose terminator slot has not been assigned.
    pub fn value_at(&self, node: usize) -> Option<i32> {
        #[cfg(feature = "reduced-trie")]
        {
            // In reduced-trie mode a non-negative `base_` directly
            // encodes the value.
            if self.array[node].base_ >= 0 {
                return Some(self.array[node].base_);
            }
        }
        let base = self.array[node].base();
        if base < 0 {
            return None;
        }
        let value_slot = base as usize;
        let n = self.array.get(value_slot)?;
        if n.check == node as i32 {
            Some(n.base_)
        } else {
            None
        }
    }

    /// Iterate `(byte, child_node)` pairs for the real edges out of
    /// `node`, skipping the terminator (byte 0) used to store values.
    ///
    /// Walks the cedar sibling chain in O(fan-out).  Defensive on
    /// `check` ownership so it stays sound even if called on a free
    /// or partially-built slot.
    #[inline]
    pub fn children_iter(&self, node: usize) -> ChildIter<'_> {
        // The first physical child byte; 0 means the only "child" is
        // the terminator value slot (advance to its sibling).
        let first = self.n_infos[node].child;
        let start_byte = if first == 0 {
            let base = self.array[node].base();
            if base < 0 {
                0
            } else {
                self.n_infos[base as usize].sibling
            }
        } else {
            first
        };
        ChildIter {
            cedar: self,
            node,
            next_byte: start_byte,
        }
    }

    /// Follow the edge labelled `byte` out of `node`, returning the
    /// child node index if such an edge exists.
    ///
    /// This is the constant-time random-access primitive that custom
    /// walks (Levenshtein automaton intersection, regex/wildcard DFA
    /// product walks, prefix predicates) rely on.  Together with
    /// [`Cedar::root`], [`Cedar::value_at`] and [`Cedar::children_iter`]
    /// it forms a complete, allocation-free read-only API over the
    /// double-array trie.
    ///
    /// Rejects `byte == 0` since byte 0 is reserved internally as the
    /// terminator slot for values — callers must read those via
    /// [`Cedar::value_at`] rather than as an edge transition.
    #[inline]
    pub fn transition(&self, node: usize, byte: u8) -> Option<usize> {
        if byte == 0 {
            return None;
        }
        let base = self.array.get(node)?.base();
        if base < 0 {
            return None;
        }
        let child = (base ^ (byte as i32)) as usize;
        let n = self.array.get(child)?;
        if n.check == node as i32 {
            Some(child)
        } else {
            None
        }
    }

    /// Number of real outgoing edges from `node` (terminator excluded).
    ///
    /// Convenience over `children_iter(node).count()` with identical
    /// O(fan-out) cost — provided so callers can quickly distinguish
    /// branch nodes from value-only leaves without constructing an
    /// iterator.
    #[inline]
    pub fn num_children(&self, node: usize) -> usize {
        self.children_iter(node).count()
    }

    /// `true` if `node` has no real outgoing edges (terminator excluded).
    ///
    /// A leaf may still carry a value — combine with [`Cedar::value_at`]
    /// to distinguish value-leaves from genuinely empty slots.
    #[inline]
    pub fn is_leaf(&self, node: usize) -> bool {
        self.children_iter(node).next().is_none()
    }

    // To get the cursor of the first leaf node starting by `from`
    fn begin(&self, mut from: usize, mut p: usize) -> (Option<i32>, usize, usize) {
        let base = self.array[from].base();
        let mut c = self.n_infos[from].child;

        if from == 0 {
            c = self.n_infos[(base ^ (c as i32)) as usize].sibling;

            // if no sibling couldn be found from the virtual root, then we are done.
            if c == 0 {
                return (None, from, p);
            }
        }

        // recursively traversing down to look for the first leaf.
        while c != 0 {
            from = (self.array[from].base() ^ (c as i32)) as usize;
            c = self.n_infos[from].child;
            p += 1;
        }

        #[cfg(feature = "reduced-trie")]
        {
            if self.array[from].base_ >= 0 {
                return (Some(self.array[from].base_), from, p);
            }
        }

        // To return the value of the leaf.
        let v = self.array[(self.array[from].base() ^ (c as i32)) as usize].base_;
        (Some(v), from, p)
    }

    // To move the cursor from one leaf to the next for the common_prefix_predict.
    fn next(&self, mut from: usize, mut p: usize, root: usize) -> (Option<i32>, usize, usize) {
        #[allow(unused_assignments)]
        let mut c: u8 = 0;

        #[cfg(feature = "reduced-trie")]
        {
            if self.array[from].base_ < 0 {
                c = self.n_infos[(self.array[from].base()) as usize].sibling;
            }
        }
        #[cfg(not(feature = "reduced-trie"))]
        {
            c = self.n_infos[(self.array[from].base()) as usize].sibling;
        }

        // traversing up until there is a sibling or it has reached the root.
        while c == 0 && from != root {
            c = self.n_infos[from as usize].sibling;
            from = self.array[from as usize].check as usize;

            p -= 1;
        }

        if c != 0 {
            // it has a sibling so we leverage on `begin` to traverse the subtree down again.
            from = (self.array[from].base() ^ (c as i32)) as usize;
            let (v_, from_, p_) = self.begin(from, p + 1);
            (v_, from_, p_)
        } else {
            // no more work since we couldn't find anything.
            (None, from, p)
        }
    }

    // pop a block at idx from the linked-list of type `from`, specially handled if it is the last
    // one in the linked-list.
    fn pop_block(&mut self, idx: i32, from: BlockType, last: bool) {
        let head: &mut i32 = match from {
            BlockType::Open => &mut self.blocks_head_open,
            BlockType::Closed => &mut self.blocks_head_closed,
            BlockType::Full => &mut self.blocks_head_full,
        };

        if last {
            *head = 0;
        } else {
            let b = self.blocks[idx as usize].clone();
            self.blocks[b.prev as usize].next = b.next;
            self.blocks[b.next as usize].prev = b.prev;

            if idx == *head {
                *head = b.next;
            }
        }
    }

    // return the block at idx to the linked-list of `to`, specially handled if the linked-list is
    // empty
    fn push_block(&mut self, idx: i32, to: BlockType, empty: bool) {
        let head: &mut i32 = match to {
            BlockType::Open => &mut self.blocks_head_open,
            BlockType::Closed => &mut self.blocks_head_closed,
            BlockType::Full => &mut self.blocks_head_full,
        };

        if empty {
            self.blocks[idx as usize].next = idx;
            self.blocks[idx as usize].prev = idx;
            *head = idx;
        } else {
            self.blocks[idx as usize].prev = self.blocks[*head as usize].prev;
            self.blocks[idx as usize].next = *head;

            let t = self.blocks[*head as usize].prev;
            self.blocks[t as usize].next = idx;
            self.blocks[*head as usize].prev = idx;
            *head = idx;
        }
    }

    /// Reallocate more spaces so that we have more free blocks.
    fn add_block(&mut self) -> i32 {
        if self.size == self.capacity {
            self.capacity += self.capacity;

            self.array.resize(self.capacity, Default::default());
            self.n_infos.resize(self.capacity, Default::default());
            self.blocks.resize(self.capacity >> 8, Block::new());
        }

        self.blocks[self.size >> 8].e_head = self.size as i32;

        // make it a doubley linked list
        self.array[self.size] = Node {
            base_: -((self.size as i32) + 255),
            check: -((self.size as i32) + 1),
        };

        for i in (self.size + 1)..(self.size + 255) {
            self.array[i] = Node {
                base_: -(i as i32 - 1),
                check: -(i as i32 + 1),
            };
        }

        self.array[self.size + 255] = Node {
            base_: -((self.size as i32) + 254),
            check: -(self.size as i32),
        };

        let is_empty = self.blocks_head_open == 0;
        let idx = (self.size >> 8) as i32;
        debug_assert!(self.blocks[idx as usize].num > 1);
        self.push_block(idx, BlockType::Open, is_empty);

        self.size += 256;

        ((self.size >> 8) - 1) as i32
    }

    // transfer the block at idx from the linked-list of `from` to the linked-list of `to`,
    // specially handle the case where the destination linked-list is empty.
    fn transfer_block(&mut self, idx: i32, from: BlockType, to: BlockType, to_block_empty: bool) {
        let is_last = idx == self.blocks[idx as usize].next; //it's the last one if the next points to itself
        let is_empty = to_block_empty && (self.blocks[idx as usize].num != 0);

        self.pop_block(idx, from, is_last);
        self.push_block(idx, to, is_empty);
    }

    /// Mark an edge `e` as used in a trie node.
    fn pop_e_node(&mut self, base: i32, label: u8, from: i32) -> i32 {
        let e: i32 = if base < 0 {
            self.find_place()
        } else {
            base ^ (label as i32)
        };

        let idx = e >> 8;
        let n = self.array[e as usize].clone();

        self.blocks[idx as usize].num -= 1;
        // move the block at idx to the correct linked-list depending the free slots it still have.
        if self.blocks[idx as usize].num == 0 {
            if idx != 0 {
                self.transfer_block(idx, BlockType::Closed, BlockType::Full, self.blocks_head_full == 0);
            }
        } else {
            self.array[(-n.base_) as usize].check = n.check;
            self.array[(-n.check) as usize].base_ = n.base_;

            if e == self.blocks[idx as usize].e_head {
                self.blocks[idx as usize].e_head = -n.check;
            }

            if idx != 0 && self.blocks[idx as usize].num == 1 && self.blocks[idx as usize].trial != self.max_trial {
                self.transfer_block(idx, BlockType::Open, BlockType::Closed, self.blocks_head_closed == 0);
            }
        }

        #[cfg(feature = "reduced-trie")]
        {
            self.array[e as usize].base_ = CEDAR_VALUE_LIMIT;
            self.array[e as usize].check = from;
            if base < 0 {
                self.array[from as usize].base_ = -(e ^ (label as i32)) - 1;
            }
        }

        #[cfg(not(feature = "reduced-trie"))]
        {
            if label != 0 {
                self.array[e as usize].base_ = -1;
            } else {
                self.array[e as usize].base_ = 0;
            }
            self.array[e as usize].check = from;
            if base < 0 {
                self.array[from as usize].base_ = e ^ (label as i32);
            }
        }

        e
    }

    /// Mark an edge `e` as free in a trie node.
    fn push_e_node(&mut self, e: i32) {
        let idx = e >> 8;
        self.blocks[idx as usize].num += 1;

        if self.blocks[idx as usize].num == 1 {
            self.blocks[idx as usize].e_head = e;
            self.array[e as usize] = Node { base_: -e, check: -e };

            if idx != 0 {
                // Move the block from 'Full' to 'Closed' since it has one free slot now.
                self.transfer_block(idx, BlockType::Full, BlockType::Closed, self.blocks_head_closed == 0);
            }
        } else {
            let prev = self.blocks[idx as usize].e_head;

            let next = -self.array[prev as usize].check;

            // Insert to the edge immediately after the e_head
            self.array[e as usize] = Node {
                base_: -prev,
                check: -next,
            };

            self.array[prev as usize].check = -e;
            self.array[next as usize].base_ = -e;

            // Move the block from 'Closed' to 'Open' since it has more than one free slot now.
            if self.blocks[idx as usize].num == 2 || self.blocks[idx as usize].trial == self.max_trial {
                debug_assert!(self.blocks[idx as usize].num > 1);
                if idx != 0 {
                    self.transfer_block(idx, BlockType::Closed, BlockType::Open, self.blocks_head_open == 0);
                }
            }

            // Reset the trial stats
            self.blocks[idx as usize].trial = 0;
        }

        if self.blocks[idx as usize].reject < self.reject[self.blocks[idx as usize].num as usize] {
            self.blocks[idx as usize].reject = self.reject[self.blocks[idx as usize].num as usize];
        }

        self.n_infos[e as usize] = Default::default();
    }

    // push the `label` into the sibling chain
    fn push_sibling(&mut self, from: usize, base: i32, label: u8, has_child: bool) {
        let keep_order: bool = if self.ordered {
            label > self.n_infos[from].child
        } else {
            self.n_infos[from].child == 0
        };

        let sibling: u8;
        {
            let mut c: &mut u8 = &mut self.n_infos[from as usize].child;
            if has_child && keep_order {
                loop {
                    let code = *c as i32;
                    c = &mut self.n_infos[(base ^ code) as usize].sibling;

                    if !(self.ordered && (*c != 0) && (*c < label)) {
                        break;
                    }
                }
            }
            sibling = *c;

            *c = label;
        }

        self.n_infos[(base ^ (label as i32)) as usize].sibling = sibling;
    }

    // remove the `label` from the sibling chain.
    #[allow(dead_code)]
    fn pop_sibling(&mut self, from: i32, base: i32, label: u8) {
        let mut c: *mut u8 = &mut self.n_infos[from as usize].child;
        unsafe {
            while *c != label {
                let code = *c as i32;
                c = &mut self.n_infos[(base ^ code) as usize].sibling;
            }

            let code = label as i32;
            *c = self.n_infos[(base ^ code) as usize].sibling;
        }
    }

    // Loop through the siblings to see which one reached the end first, which means it is the one
    // with smaller in children size, and we should try ti relocate the smaller one.
    fn consult(&self, base_n: i32, base_p: i32, mut c_n: u8, mut c_p: u8) -> bool {
        loop {
            c_n = self.n_infos[(base_n ^ (c_n as i32)) as usize].sibling;
            c_p = self.n_infos[(base_p ^ (c_p as i32)) as usize].sibling;

            if !(c_n != 0 && c_p != 0) {
                break;
            }
        }

        c_p != 0
    }

    // Collect the list of the children, and push the label as well if it is not terminal node.
    fn set_child(&self, base: i32, mut c: u8, label: u8, not_terminal: bool) -> SmallVec<[u8; 256]> {
        let mut child: SmallVec<[u8; 256]> = SmallVec::new();

        if c == 0 {
            child.push(c);
            c = self.n_infos[(base ^ (c as i32)) as usize].sibling;
        }

        if self.ordered {
            while c != 0 && c <= label {
                child.push(c);
                c = self.n_infos[(base ^ (c as i32)) as usize].sibling;
            }
        }

        if not_terminal {
            child.push(label);
        }

        while c != 0 {
            child.push(c);
            c = self.n_infos[(base ^ (c as i32)) as usize].sibling;
        }

        child
    }

    // For the case where only one free slot is needed
    fn find_place(&mut self) -> i32 {
        if self.blocks_head_closed != 0 {
            return self.blocks[self.blocks_head_closed as usize].e_head;
        }

        if self.blocks_head_open != 0 {
            return self.blocks[self.blocks_head_open as usize].e_head;
        }

        // the block is not enough, resize it and allocate it.
        self.add_block() << 8
    }

    // For the case where multiple free slots are needed.
    fn find_places(&mut self, child: &[u8]) -> i32 {
        let mut idx = self.blocks_head_open;

        // we still have available 'Open' blocks.
        if idx != 0 {
            debug_assert!(self.blocks[idx as usize].num > 1);
            let bz = self.blocks[self.blocks_head_open as usize].prev;
            let nc = child.len() as i16;

            loop {
                // only proceed if the free slots are more than the number of children. Also, we
                // save the minimal number of attempts to fail in the `reject`, it only worths to
                // try out this block if the number of children is less than that number.
                if self.blocks[idx as usize].num >= nc && nc < self.blocks[idx as usize].reject {
                    let mut e = self.blocks[idx as usize].e_head;
                    loop {
                        let base = e ^ (child[0] as i32);

                        let mut i = 1;
                        // iterate through the children to see if they are available: (check < 0)
                        while self.array[(base ^ (child[i] as i32)) as usize].check < 0 {
                            if i == child.len() - 1 {
                                // we have found the available block.
                                self.blocks[idx as usize].e_head = e;
                                return e;
                            }
                            i += 1;
                        }

                        // we save the next free block's information in `check`
                        e = -self.array[e as usize].check;
                        if e == self.blocks[idx as usize].e_head {
                            break;
                        }
                    }
                }

                // we broke out of the loop, that means we failed. We save the information in
                // `reject` for future pruning.
                self.blocks[idx as usize].reject = nc;
                if self.blocks[idx as usize].reject < self.reject[self.blocks[idx as usize].num as usize] {
                    // put this stats into the global array of information as well.
                    self.reject[self.blocks[idx as usize].num as usize] = self.blocks[idx as usize].reject;
                }

                let idx_ = self.blocks[idx as usize].next;

                self.blocks[idx as usize].trial += 1;

                // move this block to the 'Closed' block list since it has reached the max_trial
                if self.blocks[idx as usize].trial == self.max_trial {
                    self.transfer_block(idx, BlockType::Open, BlockType::Closed, self.blocks_head_closed == 0);
                }

                // we have finsihed one round of this cyclic doubly-linked-list.
                if idx == bz {
                    break;
                }

                // going to the next in this linked list group
                idx = idx_;
            }
        }

        self.add_block() << 8
    }

    // resolve the conflict by moving one of the the nodes to a free block.
    fn resolve(&mut self, mut from_n: usize, base_n: i32, label_n: u8) -> i32 {
        let to_pn = base_n ^ (label_n as i32);

        // the `base` and `from` for the conflicting one.
        let from_p = self.array[to_pn as usize].check;
        let base_p = self.array[from_p as usize].base();

        // whether to replace siblings of newly added
        let flag = self.consult(
            base_n,
            base_p,
            self.n_infos[from_n as usize].child,
            self.n_infos[from_p as usize].child,
        );

        // collect the list of children for the block that we are going to relocate.
        let children = if flag {
            self.set_child(base_n, self.n_infos[from_n as usize].child, label_n, true)
        } else {
            self.set_child(base_p, self.n_infos[from_p as usize].child, 255, false)
        };

        // decide which algorithm to allocate free block depending on the number of children we
        // have.
        let mut base = if children.len() == 1 {
            self.find_place()
        } else {
            self.find_places(&children)
        };

        base ^= children[0] as i32;

        let (from, base_) = if flag {
            (from_n as i32, base_n)
        } else {
            (from_p, base_p)
        };

        if flag && children[0] == label_n {
            self.n_infos[from as usize].child = label_n;
        }

        #[cfg(feature = "reduced-trie")]
        {
            self.array[from as usize].base_ = -base - 1;
        }

        #[cfg(not(feature = "reduced-trie"))]
        {
            self.array[from as usize].base_ = base;
        }

        // the actual work for relocating the chilren
        for i in 0..(children.len()) {
            let to = self.pop_e_node(base, children[i], from);
            let to_ = base_ ^ (children[i] as i32);

            if i == children.len() - 1 {
                self.n_infos[to as usize].sibling = 0;
            } else {
                self.n_infos[to as usize].sibling = children[i + 1];
            }

            if flag && to_ == to_pn {
                continue;
            }

            self.array[to as usize].base_ = self.array[to_ as usize].base_;

            #[cfg(feature = "reduced-trie")]
            let condition = self.array[to as usize].base_ < 0 && children[i] != 0;
            #[cfg(not(feature = "reduced-trie"))]
            let condition = self.array[to as usize].base_ > 0 && children[i] != 0;

            if condition {
                let mut c = self.n_infos[to_ as usize].child;

                self.n_infos[to as usize].child = c;

                loop {
                    let idx = (self.array[to as usize].base() ^ (c as i32)) as usize;
                    self.array[idx].check = to;
                    c = self.n_infos[idx].sibling;

                    if c == 0 {
                        break;
                    }
                }
            }

            if !flag && to_ == (from_n as i32) {
                from_n = to as usize;
            }

            // clean up the space that was moved away from.
            if !flag && to_ == to_pn {
                self.push_sibling(from_n, to_pn ^ (label_n as i32), label_n, true);
                self.n_infos[to_ as usize].child = 0;

                #[cfg(feature = "reduced-trie")]
                {
                    self.array[to_ as usize].base_ = CEDAR_VALUE_LIMIT;
                }

                #[cfg(not(feature = "reduced-trie"))]
                {
                    if label_n != 0 {
                        self.array[to_ as usize].base_ = -1;
                    } else {
                        self.array[to_ as usize].base_ = 0;
                    }
                }

                self.array[to_ as usize].check = from_n as i32;
            } else {
                self.push_e_node(to_);
            }
        }

        // return the position that is free now.
        if flag {
            base ^ (label_n as i32)
        } else {
            to_pn
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Alphanumeric;
    use rand::{thread_rng, Rng};
    use std::iter;

    #[test]
    fn test_insert_and_delete() {
        let dict = vec!["a"];
        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        let result = cedar.exact_match_search("ab").map(|x| x.0);
        assert_eq!(None, result);

        cedar.update("ab", 1);
        let result = cedar.exact_match_search("ab").map(|x| x.0);
        assert_eq!(Some(1), result);

        cedar.erase("ab");
        let result = cedar.exact_match_search("ab").map(|x| x.0);
        assert_eq!(None, result);

        cedar.update("abc", 2);
        let result = cedar.exact_match_search("abc").map(|x| x.0);
        assert_eq!(Some(2), result);

        cedar.erase("abc");
        let result = cedar.exact_match_search("abc").map(|x| x.0);
        assert_eq!(None, result);
    }

    #[test]
    fn test_common_prefix_search() {
        let dict = vec![
            "a",
            "ab",
            "abc",
            "アルゴリズム",
            "データ",
            "構造",
            "网",
            "网球",
            "网球拍",
            "中",
            "中华",
            "中华人民",
            "中华人民共和国",
        ];
        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        let result: Vec<i32> = cedar
            .common_prefix_search("abcdefg")
            .unwrap()
            .iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(vec![0, 1, 2], result);

        let result: Vec<i32> = cedar
            .common_prefix_search("网球拍卖会")
            .unwrap()
            .iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(vec![6, 7, 8], result);

        let result: Vec<i32> = cedar
            .common_prefix_search("中华人民共和国")
            .unwrap()
            .iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(vec![9, 10, 11, 12], result);

        let result: Vec<i32> = cedar
            .common_prefix_search("データ構造とアルゴリズム")
            .unwrap()
            .iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(vec![4], result);
    }

    #[test]
    fn test_common_prefix_iter() {
        let dict = vec![
            "a",
            "ab",
            "abc",
            "アルゴリズム",
            "データ",
            "構造",
            "网",
            "网球",
            "网球拍",
            "中",
            "中华",
            "中华人民",
            "中华人民共和国",
        ];

        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        let result: Vec<i32> = cedar.common_prefix_iter("abcdefg").map(|x| x.0).collect();
        assert_eq!(vec![0, 1, 2], result);

        let result: Vec<i32> = cedar.common_prefix_iter("网球拍卖会").map(|x| x.0).collect();
        assert_eq!(vec![6, 7, 8], result);

        let result: Vec<i32> = cedar.common_prefix_iter("中华人民共和国").map(|x| x.0).collect();
        assert_eq!(vec![9, 10, 11, 12], result);

        let result: Vec<i32> = cedar
            .common_prefix_iter("データ構造とアルゴリズム")
            .map(|x| x.0)
            .collect();
        assert_eq!(vec![4], result);
    }

    #[test]
    fn test_common_prefix_predict() {
        let dict = vec!["a", "ab", "abc"];
        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        let result: Vec<i32> = cedar.common_prefix_predict("a").unwrap().iter().map(|x| x.0).collect();
        assert_eq!(vec![0, 1, 2], result);
    }

    #[test]
    fn test_exact_match_search() {
        let dict = vec!["a", "ab", "abc"];
        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        let result = cedar.exact_match_search("abc").map(|x| x.0);
        assert_eq!(Some(2), result);
    }

    #[test]
    fn test_unicode_han_sip() {
        let dict = vec!["讥䶯䶰", "讥䶯䶰䶱䶲", "讥䶯䶰䶱䶲䶳䶴䶵𦡦"];

        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        let result: Vec<i32> = cedar.common_prefix_iter("讥䶯䶰䶱䶲䶳䶴䶵𦡦").map(|x| x.0).collect();
        assert_eq!(vec![0, 1, 2], result);
    }

    #[test]
    fn test_unicode_grapheme_cluster() {
        let dict = vec!["a", "abc", "abcde\u{0301}"];

        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        let result: Vec<i32> = cedar
            .common_prefix_iter("abcde\u{0301}\u{1100}\u{1161}\u{AC00}")
            .map(|x| x.0)
            .collect();
        assert_eq!(vec![0, 1, 2], result);
    }

    #[test]
    fn test_erase() {
        let dict = vec!["a", "ab", "abc"];
        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        cedar.erase("abc");
        assert!(cedar.exact_match_search("abc").is_none());
        assert!(cedar.exact_match_search("ab").is_some());
        assert!(cedar.exact_match_search("a").is_some());

        cedar.erase("ab");
        assert!(cedar.exact_match_search("ab").is_none());
        assert!(cedar.exact_match_search("a").is_some());

        cedar.erase("a");
        assert!(cedar.exact_match_search("a").is_none());
    }

    #[test]
    fn test_erase_on_internal_key() {
        let mut cedar = Cedar::new();

        cedar.update("aa", 0);
        assert!(cedar.exact_match_search("aa").is_some());
        cedar.update("ab", 1);
        assert!(cedar.exact_match_search("ab").is_some());

        cedar.erase("a");
        assert!(cedar.exact_match_search("a").is_none());
        cedar.erase("aa");
        assert!(cedar.exact_match_search("aa").is_none());
        cedar.erase("ab");
        assert!(cedar.exact_match_search("ab").is_none());
    }

    #[test]
    fn test_update() {
        let dict = vec!["a", "ab", "abc"];
        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        cedar.update("abcd", 3);

        assert!(cedar.exact_match_search("a").is_some());
        assert!(cedar.exact_match_search("ab").is_some());
        assert!(cedar.exact_match_search("abc").is_some());
        assert!(cedar.exact_match_search("abcd").is_some());
        assert!(cedar.exact_match_search("abcde").is_none());

        let dict = vec!["a", "ab", "abc"];
        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);
        cedar.update("bachelor", 1);
        cedar.update("jar", 2);
        cedar.update("badge", 3);
        cedar.update("baby", 4);

        assert!(cedar.exact_match_search("bachelor").is_some());
        assert!(cedar.exact_match_search("jar").is_some());
        assert!(cedar.exact_match_search("badge").is_some());
        assert!(cedar.exact_match_search("baby").is_some());
        assert!(cedar.exact_match_search("abcde").is_none());

        let dict = vec!["a", "ab", "abc"];
        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);
        cedar.update("中", 1);
        cedar.update("中华", 2);
        cedar.update("中华人民", 3);
        cedar.update("中华人民共和国", 4);

        assert!(cedar.exact_match_search("中").is_some());
        assert!(cedar.exact_match_search("中华").is_some());
        assert!(cedar.exact_match_search("中华人民").is_some());
        assert!(cedar.exact_match_search("中华人民共和国").is_some());
    }

    #[test]
    fn test_quickcheck_like() {
        let mut rng = thread_rng();
        let mut dict: Vec<String> = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let chars: Vec<u8> = iter::repeat(()).map(|()| rng.sample(Alphanumeric)).take(30).collect();
            let s = String::from_utf8(chars).unwrap();
            dict.push(s);
        }

        let key_values: Vec<(&str, i32)> = dict.iter().enumerate().map(|(k, s)| (s.as_ref(), k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        for (k, s) in dict.iter().enumerate() {
            assert_eq!(cedar.exact_match_search(s).map(|x| x.0), Some(k as i32));
        }
    }

    #[test]
    fn test_quickcheck_like_with_deep_trie() {
        let mut rng = thread_rng();
        let mut dict: Vec<String> = Vec::with_capacity(1000);
        let mut s = String::new();
        for _ in 0..1000 {
            let c: char = rng.sample(Alphanumeric) as char;
            s.push(c);
            dict.push(s.clone());
        }

        let key_values: Vec<(&str, i32)> = dict.iter().enumerate().map(|(k, s)| (s.as_ref(), k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        for (k, s) in dict.iter().enumerate() {
            assert_eq!(cedar.exact_match_search(s).map(|x| x.0), Some(k as i32));
        }
    }

    #[test]
    fn test_mass_erase() {
        let mut rng = thread_rng();
        let mut dict: Vec<String> = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let chars: Vec<u8> = iter::repeat(()).map(|()| rng.sample(Alphanumeric)).take(30).collect();
            let s = String::from_utf8(chars).unwrap();

            dict.push(s);
        }

        let key_values: Vec<(&str, i32)> = dict.iter().enumerate().map(|(k, s)| (s.as_ref(), k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        for s in dict.iter() {
            cedar.erase(s);
            assert!(cedar.exact_match_search(s).is_none());
        }
    }

    #[test]
    fn test_duplication() {
        let dict = vec!["些许端", "些須", "些须", "亜", "亝", "亞", "亞", "亞丁", "亞丁港"];
        let key_values: Vec<(&str, i32)> = dict.into_iter().enumerate().map(|(k, s)| (s, k as i32)).collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_values);

        assert_eq!(cedar.exact_match_search("亞").map(|t| t.0), Some(6));
        assert_eq!(cedar.exact_match_search("亞丁港").map(|t| t.0), Some(8));
        assert_eq!(cedar.exact_match_search("亝").map(|t| t.0), Some(4));
        assert_eq!(cedar.exact_match_search("些須").map(|t| t.0), Some(1));
    }

    // -----------------------------------------------------------------
    // Low-level traversal API: root / transition / value_at /
    // children_iter / num_children / is_leaf
    // -----------------------------------------------------------------

    /// Walk a byte slice from `root` via `transition` and return the
    /// node index where the walk ends, or `None` if any byte has no
    /// matching outgoing edge.  Helper used throughout the tests below.
    fn walk(cedar: &Cedar, key: &[u8]) -> Option<usize> {
        let mut node = cedar.root();
        for &b in key {
            node = cedar.transition(node, b)?;
        }
        Some(node)
    }

    #[test]
    fn traversal_root_is_zero() {
        let cedar = Cedar::new();
        assert_eq!(cedar.root(), 0);
    }

    #[test]
    fn traversal_transition_roundtrips_exact_match() {
        let dict: Vec<(&str, i32)> = vec![
            ("a", 0),
            ("ab", 1),
            ("abc", 2),
            ("abd", 3),
            ("b", 4),
            ("bee", 5),
            ("zebra", 6),
        ];
        let mut cedar = Cedar::new();
        cedar.build(&dict);

        for (key, value) in &dict {
            let node = walk(&cedar, key.as_bytes())
                .unwrap_or_else(|| panic!("transition walk for {key:?} returned None"));
            assert_eq!(
                cedar.value_at(node),
                Some(*value),
                "value_at after transition walk for {key:?}"
            );
            // Cross-check against the existing exact_match_search.
            assert_eq!(
                cedar.exact_match_search(key).map(|t| t.0),
                Some(*value),
                "exact_match_search disagrees with transition walk for {key:?}"
            );
        }
    }

    #[test]
    fn traversal_transition_rejects_terminator_byte() {
        let mut cedar = Cedar::new();
        cedar.build(&[("a", 1i32)]);
        // Byte 0 must never be returned as an edge transition — it is
        // reserved for the value slot under each node.
        assert_eq!(cedar.transition(cedar.root(), 0), None);
    }

    #[test]
    fn traversal_transition_returns_none_for_missing_edge() {
        let mut cedar = Cedar::new();
        cedar.build(&[("abc", 1i32)]);
        let a = cedar.transition(cedar.root(), b'a').expect("'a' present");
        // No edge labelled 'z' under 'a'.
        assert_eq!(cedar.transition(a, b'z'), None);
        // No edge labelled 'q' from root.
        assert_eq!(cedar.transition(cedar.root(), b'q'), None);
    }

    #[test]
    fn traversal_value_at_is_none_on_interior_branch() {
        let mut cedar = Cedar::new();
        // "ab" is *not* inserted — only "abc" is.  Walking to "ab"
        // must land on an interior branch node with no value.
        cedar.build(&[("abc", 42i32)]);
        let ab = walk(&cedar, b"ab").expect("ab interior must exist");
        assert_eq!(cedar.value_at(ab), None);
        let abc = walk(&cedar, b"abc").expect("abc leaf must exist");
        assert_eq!(cedar.value_at(abc), Some(42));
    }

    #[test]
    fn traversal_value_at_on_branch_with_value() {
        // "ab" is *both* a value and a branch (prefix of "abc").  In
        // cedar this means the node has a terminator child *and* a
        // letter child.  value_at must return the terminator's value.
        let mut cedar = Cedar::new();
        cedar.build(&[("ab", 1i32), ("abc", 2)]);
        let ab = walk(&cedar, b"ab").expect("ab must exist");
        let abc = walk(&cedar, b"abc").expect("abc must exist");
        assert_eq!(cedar.value_at(ab), Some(1));
        assert_eq!(cedar.value_at(abc), Some(2));
    }

    #[test]
    fn traversal_children_iter_at_root_matches_distinct_first_bytes() {
        let dict: &[(&str, i32)] = &[
            ("apple", 0),
            ("ape", 1),
            ("banana", 2),
            ("car", 3),
            ("cat", 4),
            ("zebra", 5),
        ];
        let mut cedar = Cedar::new();
        cedar.build(dict);

        let mut got: Vec<u8> = cedar.children_iter(cedar.root()).map(|(b, _)| b).collect();
        got.sort_unstable();
        let mut want: Vec<u8> = dict.iter().map(|(k, _)| k.as_bytes()[0]).collect();
        want.sort_unstable();
        want.dedup();
        assert_eq!(got, want, "root children must equal distinct first bytes");
    }

    #[test]
    fn traversal_children_iter_skips_terminator() {
        // Node "ab" has BOTH a value (terminator at byte 0) AND a real
        // child 'c'.  children_iter must NOT yield byte 0, only 'c'.
        let mut cedar = Cedar::new();
        cedar.build(&[("ab", 1i32), ("abc", 2)]);
        let ab = walk(&cedar, b"ab").expect("ab");
        let children: Vec<u8> = cedar.children_iter(ab).map(|(b, _)| b).collect();
        assert_eq!(children, vec![b'c']);
        // And the (byte, child) pair indeed walks to "abc".
        let (_, c_node) = cedar.children_iter(ab).next().unwrap();
        assert_eq!(cedar.value_at(c_node), Some(2));
    }

    #[test]
    fn traversal_children_iter_full_dfs_matches_inserted_keys() {
        // Reconstruct every (key, value) pair by exhaustive DFS via
        // root / children_iter / value_at, then compare against the
        // input set.  This is the strongest invariant: if DFS round-
        // trips, the low-level API is internally consistent.
        let dict: &[(&str, i32)] = &[
            ("a", 0),
            ("ab", 1),
            ("abc", 2),
            ("abcd", 3),
            ("abd", 4),
            ("b", 5),
            ("bee", 6),
            ("zebra", 7),
            ("中", 8),
            ("中华", 9),
            ("中华人民", 10),
            ("网", 11),
            ("网球", 12),
        ];
        let mut cedar = Cedar::new();
        cedar.build(dict);

        let mut found: Vec<(Vec<u8>, i32)> = Vec::new();
        let mut stack: Vec<(usize, Vec<u8>)> = vec![(cedar.root(), Vec::new())];
        while let Some((node, prefix)) = stack.pop() {
            if let Some(v) = cedar.value_at(node) {
                // Root itself has no inserted empty key in this dict,
                // but record any value we see — the comparison below
                // will catch spurious entries.
                if !prefix.is_empty() {
                    found.push((prefix.clone(), v));
                }
            }
            for (b, child) in cedar.children_iter(node) {
                let mut next = prefix.clone();
                next.push(b);
                stack.push((child, next));
            }
        }
        found.sort();

        let mut want: Vec<(Vec<u8>, i32)> = dict
            .iter()
            .map(|(k, v)| (k.as_bytes().to_vec(), *v))
            .collect();
        want.sort();

        assert_eq!(found, want, "DFS via low-level API must round-trip all keys");
    }

    #[test]
    fn traversal_num_children_and_is_leaf() {
        let mut cedar = Cedar::new();
        cedar.build(&[("ab", 1i32), ("abc", 2), ("abd", 3), ("z", 4)]);
        let root = cedar.root();
        // Root has children {'a', 'z'}.
        assert_eq!(cedar.num_children(root), 2);
        assert!(!cedar.is_leaf(root));

        let a = cedar.transition(root, b'a').expect("a");
        let ab = cedar.transition(a, b'b').expect("ab");
        // 'ab' branches to 'c' and 'd' (terminator excluded).
        assert_eq!(cedar.num_children(ab), 2);
        assert!(!cedar.is_leaf(ab));

        let abc = cedar.transition(ab, b'c').expect("abc");
        assert_eq!(cedar.num_children(abc), 0);
        assert!(cedar.is_leaf(abc));
        // Leaf still carries a value.
        assert_eq!(cedar.value_at(abc), Some(2));

        let z = cedar.transition(root, b'z').expect("z");
        assert_eq!(cedar.num_children(z), 0);
        assert!(cedar.is_leaf(z));
        assert_eq!(cedar.value_at(z), Some(4));
    }

    #[test]
    fn traversal_handles_full_byte_range() {
        // Cover every non-zero byte 1..=255 as a single-byte key
        // under root, then verify transition + value_at round-trip
        // and children_iter enumerates the full set.
        let mut keys: Vec<(Vec<u8>, i32)> = (1u16..=255)
            .map(|b| (vec![b as u8], b as i32))
            .collect();
        let key_refs: Vec<(&str, i32)> = keys
            .iter_mut()
            .filter_map(|(k, v)| {
                // Cedar's build wants &str; skip bytes that don't form
                // valid UTF-8 on their own (>= 0x80).  We still get
                // 1..0x80 (127 keys) which is plenty to stress the
                // sibling chain at root.
                core::str::from_utf8(k).ok().map(|s| (s, *v))
            })
            .collect();
        let mut cedar = Cedar::new();
        cedar.build(&key_refs);

        for (s, v) in &key_refs {
            let n = cedar.transition(cedar.root(), s.as_bytes()[0]).unwrap();
            assert_eq!(cedar.value_at(n), Some(*v));
        }

        let mut got: Vec<u8> = cedar.children_iter(cedar.root()).map(|(b, _)| b).collect();
        got.sort_unstable();
        let mut want: Vec<u8> = key_refs.iter().map(|(s, _)| s.as_bytes()[0]).collect();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn traversal_on_empty_cedar_is_safe() {
        let cedar = Cedar::new();
        let root = cedar.root();
        assert_eq!(cedar.root(), 0);
        assert_eq!(cedar.value_at(root), None);
        assert_eq!(cedar.transition(root, b'a'), None);
        assert_eq!(cedar.num_children(root), 0);
        assert!(cedar.is_leaf(root));
        assert!(cedar.children_iter(root).next().is_none());
    }

    #[test]
    fn traversal_consistent_with_common_prefix_search() {
        // Build, then for every inserted key drive a transition walk
        // and collect ancestor values — the resulting set must equal
        // common_prefix_search(key).
        let dict: &[(&str, i32)] = &[
            ("a", 0),
            ("ab", 1),
            ("abc", 2),
            ("abcd", 3),
            ("abcde", 4),
        ];
        let mut cedar = Cedar::new();
        cedar.build(dict);

        for (key, _) in dict {
            let mut node = cedar.root();
            let mut walk_values: Vec<i32> = Vec::new();
            for &b in key.as_bytes() {
                node = match cedar.transition(node, b) {
                    Some(n) => n,
                    None => break,
                };
                if let Some(v) = cedar.value_at(node) {
                    walk_values.push(v);
                }
            }
            let mut prefix_values: Vec<i32> = cedar
                .common_prefix_search(key)
                .unwrap_or_default()
                .into_iter()
                .map(|(v, _)| v)
                .collect();
            walk_values.sort_unstable();
            prefix_values.sort_unstable();
            assert_eq!(
                walk_values, prefix_values,
                "transition+value_at walk must agree with common_prefix_search for {key:?}"
            );
        }
    }
}
