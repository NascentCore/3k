import localforage from 'localforage';

localforage.config({
  driver: localforage.INDEXEDDB,
  name: 'qanything-chat-db',
});

const storage = {
  getItem: async (key) => {
    try {
      const value = await localforage.getItem(key);
      return value;
    } catch (err) {
      console.error('Error getting item from localForage', err);
      return null;
    }
  },

  setItem: async (key, value) => {
    try {
      await localforage.setItem(key, value);
      console.log(`Item with key "${key}" saved.`);
    } catch (err) {
      console.error('Error setting item in localForage', err);
    }
  },

  removeItem: async (key) => {
    try {
      await localforage.removeItem(key);
      console.log(`Item with key "${key}" removed.`);
    } catch (err) {
      console.error('Error removing item from localForage', err);
    }
  },

  clear: async () => {
    try {
      await localforage.clear();
      console.log('All items cleared from localForage.');
    } catch (err) {
      console.error('Error clearing localForage', err);
    }
  },

  keys: async () => {
    try {
      const keys = await localforage.keys();
      return keys;
    } catch (err) {
      console.error('Error getting keys from localForage', err);
      return [];
    }
  },

  containsKey: async (key) => {
    try {
      const value = await localforage.getItem(key);
      return value !== null;
    } catch (err) {
      console.error('Error checking if key exists in localForage', err);
      return false;
    }
  },
};

export default storage;
