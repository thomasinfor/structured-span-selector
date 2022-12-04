# Required Informations
```
mention_doc, input_ids, input_mask, sentence_map, is_training, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None
```

# Gold Stuffs
```
gold_starts, gold_ends, gold_mention_cluster_map

n = number of tagged clusters
gold_starts              (1d array with len n) = start index of all tagged spans
gold_ends                (1d array with len n) = end   index of all tagged spans
gold_mention_cluster_map (1d array with len n) = cluster_id of each tagged span
```

# loss function
line 406

mp_loss = L1 (?)
....... = L2 (?)

# CFGMentionProposer (outside_mp.py)
```
INPUT:
spans                 = [[s, e], [s, e], [s, e], ...]
candidate_mention_parsing_scores = [NN(span_embedding)]
candidate_mask        = [1,1,1,1,1,1,...] (?)
non_dummy_indicator   = tagged candidates
sentence_lengths
num_top_spans         = desired number of spans
flat_span_location_indices

OUTPUT:
span_marginal     = span score
top_span_indices  = indices of selected spans
top_spans         = [[s, e], [s, e], [s, e], ...]
loss              = WCFG loss
```
