import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
import glob


class IRImageConverter:
    def __init__(self, background_color: Tuple[int, int, int] = (108, 108, 108),
                 use_segmentation: bool = True):
        self.background_color = np.array(background_color)
        self.tolerance = 10
        self.use_segmentation = use_segmentation

    def remove_background(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_segmentation:
            target_mask = np.ones(image.shape[:2], dtype=bool)
            return image.copy(), target_mask

        diff = np.abs(image.astype(float) - self.background_color.astype(float))
        background_mask = np.all(diff <= self.tolerance, axis=2)

        target_mask = ~background_mask

        target_image = image.copy()
        target_image[background_mask] = [0, 0, 0]

        return target_image, target_mask

    def extract_target_histogram(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        target_pixels = gray_image[mask]
        return target_pixels

    def histogram_matching(self, source_pixels: np.ndarray, reference_pixels: np.ndarray) -> np.ndarray:
        source_hist, _ = np.histogram(source_pixels, bins=256, range=(0, 255))
        reference_hist, _ = np.histogram(reference_pixels, bins=256, range=(0, 255))

        source_cdf = np.cumsum(source_hist).astype(float)
        reference_cdf = np.cumsum(reference_hist).astype(float)

        source_cdf /= source_cdf[-1]
        reference_cdf /= reference_cdf[-1]

        lookup_table = np.zeros(256, dtype=np.uint8)

        for i in range(256):
            closest_idx = np.argmin(np.abs(reference_cdf - source_cdf[i]))
            lookup_table[i] = closest_idx

        return lookup_table

    def apply_histogram_matching(self, source_image: np.ndarray, source_mask: np.ndarray,
                                 lookup_table: np.ndarray) -> np.ndarray:
        if len(source_image.shape) == 3:
            gray_source = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_source = source_image

        matched_image = gray_source.copy()
        matched_image[source_mask] = lookup_table[gray_source[source_mask]]

        matched_rgb = cv2.cvtColor(matched_image, cv2.COLOR_GRAY2RGB)

        if self.use_segmentation:
            matched_rgb[~source_mask] = self.background_color

        return matched_rgb

    def learn_distribution(self, folder_path: str) -> np.ndarray:
        all_target_pixels = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

        for ext in image_extensions:
            image_paths = glob.glob(os.path.join(folder_path, ext))
            image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))

            for image_path in image_paths:
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        _, target_mask = self.remove_background(image)
                        target_pixels = self.extract_target_histogram(image, target_mask)

                        if len(target_pixels) > 0:
                            all_target_pixels.extend(target_pixels)
                            print(f"Processed: {os.path.basename(image_path)}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        if not all_target_pixels:
            raise ValueError(f"No valid target pixels found in {folder_path}")

        return np.array(all_target_pixels)

    def process_single_image(self, mwir_path: str, lwir_distribution: np.ndarray,
                             swir_distribution: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mwir_image = cv2.imread(mwir_path)
        mwir_image = cv2.cvtColor(mwir_image, cv2.COLOR_BGR2RGB)

        mwir_target, mwir_mask = self.remove_background(mwir_image)
        mwir_pixels = self.extract_target_histogram(mwir_target, mwir_mask)

        lwir_lookup = self.histogram_matching(mwir_pixels, lwir_distribution)
        swir_lookup = self.histogram_matching(mwir_pixels, swir_distribution)

        generated_lwir = self.apply_histogram_matching(mwir_image, mwir_mask, lwir_lookup)
        generated_swir = self.apply_histogram_matching(mwir_image, mwir_mask, swir_lookup)

        return mwir_image, generated_lwir, generated_swir

    def create_comparison_plot(self, original_mwir: np.ndarray, generated_lwir: np.ndarray,
                               generated_swir: np.ndarray, lwir_truth: Optional[np.ndarray] = None,
                               swir_truth: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None):
        num_images = 3
        if lwir_truth is not None:
            num_images += 1
        if swir_truth is not None:
            num_images += 1

        fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
        if num_images == 1:
            axes = [axes]

        plot_idx = 0

        axes[plot_idx].imshow(original_mwir)
        axes[plot_idx].set_title('Original MWIR')
        axes[plot_idx].axis('off')
        plot_idx += 1

        axes[plot_idx].imshow(generated_lwir)
        axes[plot_idx].set_title('Generated LWIR')
        axes[plot_idx].axis('off')
        plot_idx += 1

        axes[plot_idx].imshow(generated_swir)
        axes[plot_idx].set_title('Generated SWIR')
        axes[plot_idx].axis('off')
        plot_idx += 1

        if lwir_truth is not None:
            axes[plot_idx].imshow(lwir_truth)
            axes[plot_idx].set_title('Ground Truth LWIR')
            axes[plot_idx].axis('off')
            plot_idx += 1

        if swir_truth is not None:
            axes[plot_idx].imshow(swir_truth)
            axes[plot_idx].set_title('Ground Truth SWIR')
            axes[plot_idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.close()


def main():
    USE_SEGMENTATION = True

    converter = IRImageConverter(use_segmentation=USE_SEGMENTATION)

    mwir_folder = "./data/mwir"
    output_folder = "./output"

    lwir_reference_folder = "./data/reference/lwir"
    swir_reference_folder = "./data/reference/swir"

    lwir_truth_folder = "./data/truth/lwir"
    swir_truth_folder = "./data/truth/swir"

    os.makedirs(output_folder, exist_ok=True)
    lwir_output_folder = os.path.join(output_folder, "generated_lwir")
    swir_output_folder = os.path.join(output_folder, "generated_swir")
    comparison_output_folder = os.path.join(output_folder, "comparisons")

    os.makedirs(lwir_output_folder, exist_ok=True)
    os.makedirs(swir_output_folder, exist_ok=True)
    os.makedirs(comparison_output_folder, exist_ok=True)

    try:
        print(f"Segmentation mode: {'ENABLED' if USE_SEGMENTATION else 'DISABLED'}")
        print("Learning LWIR distribution...")
        lwir_distribution = converter.learn_distribution(lwir_reference_folder)
        print(f"LWIR distribution learned from {len(lwir_distribution)} pixels")

        print("Learning SWIR distribution...")
        swir_distribution = converter.learn_distribution(swir_reference_folder)
        print(f"SWIR distribution learned from {len(swir_distribution)} pixels")

        mwir_images = glob.glob(os.path.join(mwir_folder, "*.jpg"))
        mwir_images.extend(glob.glob(os.path.join(mwir_folder, "*.png")))
        mwir_images.extend(glob.glob(os.path.join(mwir_folder, "*.bmp")))

        for i, mwir_path in enumerate(mwir_images):
            print(f"Processing {os.path.basename(mwir_path)}...")

            original_mwir, generated_lwir, generated_swir = converter.process_single_image(
                mwir_path, lwir_distribution, swir_distribution
            )

            base_name = os.path.splitext(os.path.basename(mwir_path))[0]
            lwir_truth = None
            swir_truth = None

            lwir_truth_extensions = ['.jpg', '.png', '.bmp', '.tiff', '.tif']
            swir_truth_extensions = ['.jpg', '.png', '.bmp', '.tiff', '.tif']

            for ext in lwir_truth_extensions:
                lwir_truth_path = os.path.join(lwir_truth_folder, f"{base_name}{ext}")
                if os.path.exists(lwir_truth_path):
                    lwir_truth = cv2.imread(lwir_truth_path)
                    if lwir_truth is not None:
                        lwir_truth = cv2.cvtColor(lwir_truth, cv2.COLOR_BGR2RGB)
                        print(f"Found LWIR truth: {os.path.basename(lwir_truth_path)}")
                        break

            for ext in swir_truth_extensions:
                swir_truth_path = os.path.join(swir_truth_folder, f"{base_name}{ext}")
                if os.path.exists(swir_truth_path):
                    swir_truth = cv2.imread(swir_truth_path)
                    if swir_truth is not None:
                        swir_truth = cv2.cvtColor(swir_truth, cv2.COLOR_BGR2RGB)
                        print(f"Found SWIR truth: {os.path.basename(swir_truth_path)}")
                        break

            comparison_save_path = os.path.join(comparison_output_folder, f"comparison_{base_name}.png")
            converter.create_comparison_plot(
                original_mwir, generated_lwir, generated_swir,
                lwir_truth, swir_truth, comparison_save_path
            )

            cv2.imwrite(os.path.join(lwir_output_folder, f"{base_name}.jpg"),
                        cv2.cvtColor(generated_lwir, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(swir_output_folder, f"{base_name}.jpg"),
                        cv2.cvtColor(generated_swir, cv2.COLOR_RGB2BGR))

            print(f"Completed processing {i + 1}/{len(mwir_images)} images")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()