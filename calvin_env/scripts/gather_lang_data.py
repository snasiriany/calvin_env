import numpy as np
import os

dataset_path = '/home/soroushn/research/calvin/dataset/task_D_D/training'
lang_anotations = np.load(os.path.join(dataset_path, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True)[()]
start_end_ids = lang_anotations['info']['indx']
task_ann = lang_anotations['language']['task']
lang_emb = lang_anotations['language']['emb']
lang_ann = lang_anotations['language']['ann']

task_ann_to_id = {
    ann: i for (i, ann) in enumerate(sorted(list(set(task_ann))))
}

ds = {}

for seg_idx in range(len(start_end_ids)):
    start, end = start_end_ids[seg_idx]

    for t in range(start, end + 1):
        if t not in ds:
            ds[t] = {
                'task_ids': [],
                'task_anns': [],
                'lang_embs': [],
                'lang_anns': [],
            }

        ds[t]['task_ids'].append(task_ann_to_id[task_ann[seg_idx]])
        ds[t]['task_anns'].append(task_ann[seg_idx])
        ds[t]['lang_embs'].append(lang_emb[seg_idx].squeeze())
        ds[t]['lang_anns'].append(lang_ann[seg_idx])

np.save(os.path.join(dataset_path, 'lang_annotations/lang_data.npy'), ds)

