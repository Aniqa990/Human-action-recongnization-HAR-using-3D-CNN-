import json
from pathlib import Path

import torch
import torch.utils.data as data
from PIL import Image


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path, class_labels_map):
    video_ids   = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        if value['subset'] != subset:
            continue

        video_ids.append(key)
        annotations.append(value['annotations'])

        label = value['annotations']['label']

        # Try class-nested path first: root/fight/video_name
        nested_path = root_path / label / key
        flat_path   = root_path / key

        if nested_path.exists():
            video_paths.append(nested_path)
        else:
            video_paths.append(flat_path)

    return video_ids, video_paths, annotations


class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 target_type='label'):

        self.data, self.class_names = self.__make_data(
            root_path, annotation_path, subset)

        self.spatial_transform  = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform   = target_transform
        self.target_type        = target_type

    def __make_data(self, root_path, annotation_path, subset):
        with annotation_path.open('r') as f:
            data = json.load(f)

        class_labels_map = get_class_labels(data)
        class_names      = list(data['labels'])

        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, Path(''), class_labels_map)

        dataset = []
        for i in range(len(video_ids)):
            label_name = annotations[i]['label']
            if label_name not in class_labels_map:
                continue

            label   = class_labels_map[label_name]
            segment = annotations[i].get('segment', [1, 0])
            n_frames = segment[1] - segment[0] + 1

            if n_frames <= 0:
                existing = sorted(video_paths[i].glob('image_*.jpg'))
                n_frames = len(existing)

            if n_frames <= 0:
                print(f'  [warn] No frames found: {video_paths[i]}')
                continue

            dataset.append({
                'video':    video_paths[i],
                'segment':  (segment[0], segment[0] + n_frames - 1),
                'n_frames': n_frames,
                'video_id': video_ids[i],
                'label':    label,
            })

        return dataset, class_names

    def __load_clip(self, path, frame_indices):
        from torchvision import transforms as T
        to_tensor = T.ToTensor()

        clip = []
        for i in frame_indices:
            img_path = path / f'image_{i:05d}.jpg'
            try:
                if not img_path.exists():
                    existing = sorted(path.glob('image_*.jpg'))
                    if not existing:
                        raise FileNotFoundError(f'No frames in {path}')
                    img_path = existing[min(i, len(existing)) - 1]
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (112, 112), (0, 0, 0))

            if self.spatial_transform is not None:
                img = self.spatial_transform(img)
                if not isinstance(img, torch.Tensor):
                    img = to_tensor(img)
            else:
                img = to_tensor(img)

            clip.append(img)

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip

    def __getitem__(self, index):
        sample = self.data[index]
        path   = sample['video']

        frame_indices = list(range(sample['segment'][0],
                                   sample['segment'][1] + 1))

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        # flatten in case temporal_transform returns nested list
        import itertools
        if not isinstance(frame_indices, (list, tuple)):
            frame_indices = [int(frame_indices)]
        elif isinstance(frame_indices[0], (list, tuple)):
            frame_indices = [int(i) for i in itertools.chain.from_iterable(frame_indices)]
        else:
            frame_indices = [int(i) for i in frame_indices]

        clip   = self.__load_clip(path, frame_indices)
        target = sample['label']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


# ── Public API called by main.py ──────────────────────────────────────────────
def get_training_data(video_path,
                      annotation_path,
                      dataset_name,
                      input_type,
                      file_type,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None):

    dataset = VideoDataset(
        video_path,
        annotation_path,
        'training',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
    )
    return dataset


def get_validation_data(video_path,
                        annotation_path,
                        dataset_name,
                        input_type,
                        file_type,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):

    dataset = VideoDataset(
        video_path,
        annotation_path,
        'validation',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
    )
    return dataset, None


def get_inference_data(video_path,
                       annotation_path,
                       dataset_name,
                       input_type,
                       file_type,
                       subset,
                       spatial_transform=None,
                       temporal_transform=None,
                       target_transform=None):

    dataset = VideoDataset(
        video_path,
        annotation_path,
        subset,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
    )
    return dataset, None
