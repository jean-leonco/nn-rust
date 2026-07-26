use std::collections::{HashMap, VecDeque};

use crate::model::Session;

/// LRU cache of [`Session`] instances keyed by batch size.
#[derive(Debug)]
pub struct SessionCache {
    map: HashMap<usize, Session>,
    order: VecDeque<usize>,
    max: usize,
}

impl std::default::Default for SessionCache {
    fn default() -> Self {
        Self::new(None)
    }
}

impl SessionCache {
    /// Creates a cache holding at most `max` sessions. Defaults to 100.
    pub fn new(max: Option<usize>) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            max: max.unwrap_or(100),
        }
    }

    /// Returns the cached session for `batch_size`, or `None` on miss.
    pub fn get(&mut self, batch_size: usize) -> Option<&mut Session> {
        if self.map.contains_key(&batch_size) {
            self.touch(batch_size);
            self.map.get_mut(&batch_size)
        } else {
            None
        }
    }

    fn touch(&mut self, batch_size: usize) {
        if let Some(index) = self.order.iter().position(|&size| size == batch_size) {
            self.order.remove(index);
        }
        self.order.push_back(batch_size);
    }

    /// Inserts or replaces the session for `batch_size`.
    pub fn put(&mut self, batch_size: usize, session: Session) {
        if self.map.contains_key(&batch_size) {
            self.touch(batch_size);
            self.map.insert(batch_size, session);
        } else {
            if self.map.len() >= self.max
                && let Some(oldest) = self.order.pop_front()
            {
                self.map.remove(&oldest);
            }
            self.order.push_back(batch_size);
            self.map.insert(batch_size, session);
        }
    }
}
