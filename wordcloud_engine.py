from PIL import Image
import numpy as np
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from scipy.ndimage import label
import os
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageProcessor:
    def __init__(self, image_path, sample_rate=10):
        self.image_path = image_path
        self.sample_rate = sample_rate

    def load_and_process_image(self):
        img = Image.open(self.image_path).convert('RGBA')
        return np.array(img)

    def detect_color_areas(self, target_color, tolerance=10):
        img_data = self.load_and_process_image()
        img_height = img_data.shape[0]

        if DEVICE.type == "cuda":
            img_tensor = torch.from_numpy(img_data[:, :, :3].astype(np.float32)).to(DEVICE)
            target_tensor = torch.tensor(list(target_color), dtype=torch.float32, device=DEVICE)
            color_mask = (torch.abs(img_tensor - target_tensor) <= tolerance).all(dim=-1).cpu().numpy()
        else:
            color_mask = np.all(np.abs(img_data[:, :, :3] - target_color) <= tolerance, axis=-1)

        labeled_array, num_features = label(color_mask)

        regions_coords = []
        regions_masks = []
        for region_label in range(1, num_features + 1):
            region_mask = labeled_array == region_label
            x_coords, y_coords = np.where(region_mask)
            x_coords = x_coords[::self.sample_rate]
            y_coords = y_coords[::self.sample_rate]
            y_coords = img_height - y_coords
            coords = np.vstack([x_coords, y_coords]).T

            if len(coords) < 5:
                continue

            regions_coords.append(coords)
            regions_masks.append(region_mask)

        return regions_coords, regions_masks, labeled_array


class RotationAngleCalculator:
    def perform_pca(self, coords):
        if len(coords) < 2:
            return np.array([1, 0])

        swapped_coords = coords.copy()
        swapped_coords[:, [0, 1]] = swapped_coords[:, [1, 0]]

        pca = PCA(n_components=1)
        pca.fit(swapped_coords)
        return pca.components_[0]

    def calculate_angle_from_direction(self, direction):
        dx, dy = direction
        angle_deg = math.degrees(math.atan2(dy, dx))

        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        return angle_deg

    def calculate_rotation_angle(self, coords):
        return self.calculate_angle_from_direction(self.perform_pca(coords))


class WordCloudGenerator:
    def __init__(self, text, region_mask, rotation_angle, background_color, color_func=None, scale=1):
        self.text = text
        self.region_mask = region_mask
        self.rotation_angle = rotation_angle
        self.background_color = background_color
        self.color_func = color_func if color_func else self.default_color_func
        self.scale = scale
        self.original_size = (self.region_mask.shape[0] * scale, self.region_mask.shape[1] * scale)

    def default_color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        r, g, b = self.background_color
        return f"rgb({r}, {g}, {b})"

    def generate_word_cloud(self, mask_image_np_rotated):
        height, width = mask_image_np_rotated.shape[:2]
        self.word_cloud = WordCloud(
            background_color=None,
            mode='RGBA',
            repeat=True,
            mask=mask_image_np_rotated,
            prefer_horizontal=1.0,
            color_func=self.color_func,
            height=height,
            width=width,
            regexp=None,
            include_numbers=True,
            min_font_size=1,
            max_font_size=max(height, width),
            max_words=400,
        ).generate(self.text)

    def rotate_back_and_crop(self, rotated_size):
        self.rotated_wc_image = Image.fromarray(self.word_cloud.to_array()).rotate(
            self.rotation_angle, expand=True, resample=Image.BICUBIC
        )
        self.final_size = self.rotated_wc_image.size

        delta_w_1 = (rotated_size[0] - self.original_size[1]) / 2
        delta_h_1 = (rotated_size[1] - self.original_size[0]) / 2
        delta_w_2 = (self.final_size[0] - rotated_size[0]) / 2
        delta_h_2 = (self.final_size[1] - rotated_size[1]) / 2

        total_delta_w = delta_w_1 + delta_w_2
        total_delta_h = delta_h_1 + delta_h_2

        self.final_image = self.rotated_wc_image.crop((
            total_delta_w,
            total_delta_h,
            self.final_size[0] - total_delta_w,
            self.final_size[1] - total_delta_h,
        ))

    def generate_colored_word_cloud(self, mask_image_np_rotated, rotated_size):
        self.generate_word_cloud(mask_image_np_rotated)
        self.rotate_back_and_crop(rotated_size)
        return self.final_image

    def generate_black_word_cloud(self, mask_image_np_rotated, rotated_size):
        self.word_cloud.recolor(color_func=lambda *args, **kwargs: (0, 0, 0))
        self.rotate_back_and_crop(rotated_size)
        return self.final_image

    def generate_white_word_cloud(self, mask_image_np_rotated, rotated_size):
        self.word_cloud.recolor(color_func=lambda *args, **kwargs: (255, 255, 255))
        self.rotate_back_and_crop(rotated_size)
        return self.final_image

    def generate_single_black_word_cloud(self, mask_image_np_rotated, rotated_size):
        saved_layout = self.word_cloud.layout_
        self.word_cloud.layout_ = saved_layout[:1]
        self.word_cloud.recolor(color_func=lambda *args, **kwargs: (0, 0, 0))
        self.rotate_back_and_crop(rotated_size)
        self.word_cloud.layout_ = saved_layout
        return self.final_image

    def generate_single_white_word_cloud(self, mask_image_np_rotated, rotated_size):
        saved_layout = self.word_cloud.layout_
        self.word_cloud.layout_ = saved_layout[:1]
        self.word_cloud.recolor(color_func=lambda *args, **kwargs: (255, 255, 255))
        self.rotate_back_and_crop(rotated_size)
        self.word_cloud.layout_ = saved_layout
        return self.final_image


def generate_rotated_wordclouds(image_path, color_text_pairs, is_adaptive=True, output_image_prefix='Output_', render_scale=2):
    image_processor = ImageProcessor(image_path=image_path)

    input_image = Image.open(image_path).convert('RGBA')
    img_width, img_height = input_image.size

    final_image_white = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 255))
    final_image_original_black = input_image.copy()
    final_image_original_white = input_image.copy()
    final_image_single_black = input_image.copy()
    final_image_single_white = input_image.copy()

    for color, text in tqdm(color_text_pairs.items(), desc="Processing colors"):
        regions_coords, regions_masks, _ = image_processor.detect_color_areas(color)

        if not regions_coords:
            continue

        for region_coords, region_mask in zip(regions_coords, regions_masks):
            angle_calculator = RotationAngleCalculator()
            rotation_angle = (
                angle_calculator.calculate_rotation_angle(region_coords)
                if is_adaptive
                else np.random.uniform(-45, 45)
            )

            word_cloud_generator = WordCloudGenerator(
                text=text,
                region_mask=region_mask,
                rotation_angle=rotation_angle,
                background_color=color,
                scale=render_scale,
            )

            inverted_mask = np.logical_not(region_mask)
            mask_image = Image.fromarray((inverted_mask * 255).astype(np.uint8)).convert('L')
            mask_image_rotated = mask_image.rotate(-rotation_angle, expand=True, fillcolor=255)
            rotated_size = mask_image_rotated.size

            if render_scale > 1:
                mask_scaled = mask_image_rotated.resize(
                    (mask_image_rotated.width * render_scale, mask_image_rotated.height * render_scale),
                    Image.NEAREST,
                )
                mask_image_np_rotated = np.array(mask_scaled)
                rotated_size_scaled = mask_scaled.size
            else:
                mask_image_np_rotated = np.array(mask_image_rotated)
                rotated_size_scaled = rotated_size

            def _composite(base, wc_img):
                resized = wc_img.resize(base.size, Image.LANCZOS).convert('RGBA')
                return Image.alpha_composite(base, resized)

            wc_colored = word_cloud_generator.generate_colored_word_cloud(mask_image_np_rotated, rotated_size_scaled)
            final_image_white = _composite(final_image_white, wc_colored)

            wc_black = word_cloud_generator.generate_black_word_cloud(mask_image_np_rotated, rotated_size_scaled)
            final_image_original_black = _composite(final_image_original_black, wc_black)

            wc_white = word_cloud_generator.generate_white_word_cloud(mask_image_np_rotated, rotated_size_scaled)
            final_image_original_white = _composite(final_image_original_white, wc_white)

            wc_single_black = word_cloud_generator.generate_single_black_word_cloud(mask_image_np_rotated, rotated_size_scaled)
            final_image_single_black = _composite(final_image_single_black, wc_single_black)

            wc_single_white = word_cloud_generator.generate_single_white_word_cloud(mask_image_np_rotated, rotated_size_scaled)
            final_image_single_white = _composite(final_image_single_white, wc_single_white)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = ["Black", "Colored", "White", "Single (Black)", "Single (White)"]
    images = [final_image_original_black, final_image_white, final_image_original_white,
              final_image_single_black, final_image_single_white]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    final_image_original_black.save(f"{output_image_prefix}_{base_name}_black.png")
    final_image_white.save(f"{output_image_prefix}_{base_name}_colored.png")
    final_image_original_white.save(f"{output_image_prefix}_{base_name}_white.png")
    final_image_single_black.save(f"{output_image_prefix}_{base_name}_single_black.png")
    final_image_single_white.save(f"{output_image_prefix}_{base_name}_single_white.png")
    print(f"已保存 5 张结果图（前缀：{output_image_prefix}_{base_name}_）")
