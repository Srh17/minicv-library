import numpy as np
import matplotlib.pyplot as plt
import minicv as mcv


def main():
    try:
        img = mcv.read_image("test.jpg")
        gray = mcv.rgb_to_gray(img)

        norm = mcv.normalize_image(gray, mode="minmax")
        mean_blur = mcv.mean_filter(gray, kernel_size=5)
        gauss = mcv.gaussian_filter(gray, size=5, sigma=1.0)
        median = mcv.median_filter(gray, kernel_size=3)
        sobel = mcv.sobel_magnitude(gray)
        otsu = mcv.otsu_threshold(gray)
        adaptive = mcv.adaptive_threshold_mean(gray, block_size=11, c=0.03)
        equalized = mcv.equalize_histogram(gray)
        bit7 = mcv.bit_plane_slice(gray, 7)
        log_img = mcv.log_transform(gray)

        resized_nn = mcv.resize_nearest(gray, (200, 200))
        resized_bl = mcv.resize_bilinear(gray, (200, 200))
        translated = mcv.translate_image(gray, tx=30, ty=20)
        rotated = mcv.rotate_image(gray, angle=30, interpolation="bilinear")

        canvas = img.copy()
        canvas = mcv.draw_point(canvas, (30, 30), (1, 0, 0), thickness=5)
        canvas = mcv.draw_line(canvas, (20, 50), (180, 120), (0, 1, 0), thickness=2)
        canvas = mcv.draw_rectangle(canvas, (50, 60), (180, 150), (0, 0, 1), thickness=2, filled=False)
        canvas = mcv.draw_polygon(canvas, [(220, 40), (280, 80), (250, 140), (200, 100)], (1, 1, 0), thickness=2)

        hist = mcv.compute_histogram(gray)
        global_desc = mcv.global_feature_vector(gray)
        color_desc = mcv.color_moments(img)
        grad_desc1 = mcv.gradient_features(gray)
        grad_desc2 = mcv.gradient_descriptor(gray)

        print("Global descriptor:", global_desc)
        print("Color moments:", color_desc)
        print("Gradient features:", grad_desc1)
        print("Gradient descriptor:", grad_desc2)

        plt.figure(figsize=(18, 12))

        images = [
            img, gray, norm, mean_blur,
            gauss, median, sobel, equalized,
            otsu, adaptive, bit7, log_img,
            resized_nn, resized_bl, translated, rotated
        ]

        titles = [
            "Original RGB", "Grayscale", "Normalized", "Mean Filter",
            "Gaussian Filter", "Median Filter", "Sobel Magnitude", "Hist Equalization",
            "Otsu Threshold", "Adaptive Threshold", "Bit Plane 7", "Log Transform",
            "Resize Nearest", "Resize Bilinear", "Translation", "Rotation"
        ]

        for i, (im, title) in enumerate(zip(images, titles), start=1):
            plt.subplot(4, 4, i)
            if im.ndim == 2:
                plt.imshow(im, cmap="gray")
            else:
                plt.imshow(im)
            plt.title(title)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(canvas)
        plt.title("Drawing Primitives")
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(hist)
        plt.title("Histogram")
        plt.xlabel("Intensity Bin")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        mcv.save_image(canvas, "drawing_result.png")
        mcv.save_image(rotated, "rotated_result.png")
        mcv.save_image(equalized, "equalized_result.png")

        print("Pipeline executed successfully using MiniCV library.")
        print("All tests completed successfully.")
        

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()