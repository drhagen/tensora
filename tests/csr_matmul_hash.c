int evaluate(taco_tensor_t *a, taco_tensor_t *b, taco_tensor_t *c) {
  int32_t i_dim = a->dimensions[0];
  int32_t k_dim = a->dimensions[1];
  int32_t j_dim = b->dimensions[1];
  int32_t* restrict a_1_pos = (int32_t*)(a->indices[1][0]);
  int32_t* restrict a_1_crd = (int32_t*)(a->indices[1][1]);
  double* restrict a_vals = (double*)(a->vals);
  int32_t* restrict b_1_pos = (int32_t*)(b->indices[1][0]);
  int32_t* restrict b_1_crd = (int32_t*)(b->indices[1][1]);
  double* restrict b_vals = (double*)(b->vals);
  int32_t* restrict c_1_pos = (int32_t*)(c->indices[1][0]);
  int32_t* restrict c_1_crd = (int32_t*)(c->indices[1][1]);
  double* restrict c_vals = (double*)(c->vals);

  a_1_pos = (int32_t*)malloc(sizeof(int32_t) * (a->dimensions[0] + 1));
  a_1_pos[0] = 0;
  int32_t a_1_crd_capacity = 1048576;
  a_1_crd = (int32_t*)malloc(sizeof(int32_t) * a_1_crd_capacity
  int32_t p_a_0_1 = 0;
  int32_t a_vals_capacity = 1048576;
  a_vals = (double*)malloc(sizeof(double) * a_vals_capacity);

  for (int32_t i = 0; i < i_dim; i++) {
    int32_t p_b_0_0 = i;

    hash_table_t hash_table = hash_construct();

    for (int32_t p_b_0_1 = b_1_pos[p_b_0_0]; p_b_0_1 < b_1_pos[p_b_0_0+1]; p_b_0_1++) {
      int32_t i_b_0_1 = b_1_crd[p_b_0_1];
      int32_t j = i_b_0_1;
      int32_t p_c_0_0 = j;

      for (int32_t p_c_0_1 = c_1_pos[p_c_0_0]; p_c_0_1 < c_1_pos[p_c_0_0+1]; p_c_0_1++) {
        int32_t i_c_0_1 = c_1_crd[p_c_0_1];
        int32_t k = i_c_0_1;

        // Once the last sparse index is known, find the item in the hash table, possibly allocating it
        // This bucket has enough space to store the remaining dense dimensions
        double[] bucket = hash_insert(&hash_table, {k});

        // Write the dense elements into that bucket
        bucket[0] = (b_vals[p_b_0_1] * c_vals[p_c_0_1]);
      }
    }

    uint32_t[] hash_table_order = malloc(sizeof(uint32_t) * hash_table->count);
    for (uint32_t i = 0; i < hash_table->count; i++) {
        hash_table_order[i] = i;
    }

    qsort_r(hash_table_order, hash_table->count, sizeof(uint32_t), hash_comparator, &hash_table);

    for (uint32_t i_order = 0; i_order < hash_table->count; i++) {
        a_1_crd[p_a_0_1] = hash_table->keys[i_order * hash_table->n_sparse + 0];
        for (uint32_t i_bucket = 0; i_bucket < hash_table->bucket_size; i_bucket++) {
            a_vals[p_a_0_1 + i_bucket] = hash_table->values[i_order * hash_table->bucket_size + i_bucket];
        }
        p_a_0_1 = p_a_0_1 + bucket_size;
    }

    hash_reset(&hash_table);
  }

  a->indices[1][0] = (unit8_t*)a_1_pos;
  a->indices[1][1] = (unit8_t*)a_1_crd;
  a->vals = (uint8_t*)a_vals;

  return 0;
}