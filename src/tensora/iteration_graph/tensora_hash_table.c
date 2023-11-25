#include <stdio.h>

static inline uint32_t murmur_32_scramble(uint32_t k) {
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    return k;
}

uint32_t murmur3_32(uint32_t n_sparse, const uint32_t* key) {
    uint32_t h = 1;

    for (size_t i = 0; i < n_sparse; i++) {
        h ^= murmur_32_scramble(key[i]);
        h = (h << 13) | (h >> 19);
        h = h * 5 + 0xe6546b64;
    }

    h ^= n_sparse;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

struct hash_table_t {
    uint32_t n_layers;
    uint32_t dense_start;
    taco_mode_t *modes;
    uint32_t n_sparse;
    uint32_t bucket_size;

    uint32_t count;
    uint32_t table_capacity;
    uint32_t entries_capacity;
    int32_t[] indexes;
    uint32_t[] keys;
    double[] values;
};

void hash_construct(
    hash_table_t *hash_table,
    uint32_t n_layers,
    uint32_t dense_start,
    taco_mode_t *modes,
    uint32_t *dimensions
) {
    uint32_t n_sparse = dense_start - n_layers

    uint32_t bucket_size = 1;
    for (uint32_t i = dense_start; i < n_layers; i++) {
        bucket_size *= dimensions[i];
    }

    uint32_t table_capacity = 10;  // 1 MB
    uint32_t entries_capacity = 1024*1024;
    uint32_t[] indexes = malloc(sizeof(uint32_t) * (1 << table_capacity));
    for (uint32_t i = 0; i < (1 << table_capacity); i++) {
        indexes[i] = -1;
    }

    hash_table->n_layers = n_layers;
    hash_table->dense_start = dense_start;
    hash_table->modes = modes;
    hash_table->n_sparse = n_sparse;
    hash_table->bucket_size = bucket_size;

    hash_table->count = 0;
    hash_table->table_capacity = table_capacity;
    hash_table->entries_capacity = entries_capacity;
    hash_table->indexes = indexes;
    hash_table->keys = malloc(sizeof(uint32_t) * n_sparse * entries_capacity);
    hash_table->values = malloc(sizeof(double) * bucket_size * entries_capacity);
}

void hash_realloc(hash_table_t *hash_table, uint32_t index) {
    // Heuristic to expand hash table when it is two thirds full
    if (index * 3 > (1 << hash_table->table_capacity) * 2) {
        hash_table->table_capacity++;
        free(hash_table->indexes);

        // Fill the hash table with the sentinel
        hash_table->indexes = malloc(sizeof(uint32_t) * (1 << hash_table->table_capacity));
        for (uint32_t i = 0; i < (1 << table_capacity); i++) {
            hash_table->indexes[i] = -1;
        }

        // Reinsert all the locations of elements into the hash table
        uint32_t mask = 0xffffffffu >> (32 - hash_table->table_capacity);
        for (uint32_t i = 0; i < hash_table->count; i++) {
            uint32_t hash_value = murmur3_32(hash_table->n_sparse, key);
            uint32_t short_hash = hash_value & mask;

            hash_table->indexes[short_hash] = location;
        }
    }

    if (index >= hash_table->entries_capacity) {
        uint32_t entries_capacity = max(hash_table->entries_capacity * 2, index)
        hash_table->entries_capacity = entries_capacity;
        hash_table->keys = realloc(hash_table->keys, sizeof(uint32_t) * hash_table->n_sparse * entries_capacity);
        hash_table->values = realloc(hash_table->values, sizeof(double) * hash_table->bucket_size * entries_capacity);
    }
}

double[] hash_get_bucket(hash_table_t *hash_table, uint32_t *key) {
    uint32_t hash_value = murmur3_32(hash_table->n_sparse, key);
    uint32_t mask = 0xffffffffu >> (32 - hash_table->table_capacity);
    uint32_t short_hash = hash_value & mask;

    for (;;) {
        if (hash_table->indexes[short_hash] == -1) {
            // Empty location found. Store the key and initialize the bucket.
            uint32_t location = hash_table->count;

            // Allocate more space, if needed
            hash_realloc(hash_table*, location);

            hash_table->indexes[short_hash] = location;

            for (uint32_t i = 0; i < hash_table->n_sparse; i++) {
                hash_table->keys[hash_table->n_sparse * location + i] = key[i];
            }

            double[] bucket = hash_table->values + location * hash_table->bucket_size;
            for (uint32_t i = 0; i < bucket_size; i++) {
                bucket[i] = 0.0;
            }

            hash_table->count++;
            return bucket;
        } else {
            // Location is occupied
            uint32_t location = hash_table->indexes[short_hash];

            for (uint32_t i = 0; i < hash_table->n_sparse; i++) {
                if (hash_table->keys[hash_table->n_sparse * count + i] != key[i]) {
                    // Location was filled with different key. Increment (mod table capacity) and continue.
                    short_hash = (short_hash + 1) & mask;
                    continue;
                }
            }
            // Location was filled with this key already. Return the bucket.
            return hash_table->values + location * hash_table->bucket_size
        }
    }
}

void hash_destruct(hash_table_t *hash_table) {
    free(hash_table->keys);
    free(hash_table->values);
}

int hash_comparator(uint32_t *left, uint32_t *right, hash_table_t *hash_table) {
    // Keys cannot be equal so that case can be ignored

    left_key = hash_table->keys[*left * hash_table->n_sparse];
    right_key = hash_table->keys[*right * hash_table->n_sparse];
    for (uint32_t i = 0; i < hash_table->n_sparse) {
        if (left_key[i] > right_key[i]) {
            return 1;
        }
    }
    return -1;
}
