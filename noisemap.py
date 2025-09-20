from PIL import Image
import math

def create_image_from_file(input_file, output_image):
    # Read numbers from the text file
    with open(input_file, 'r') as f:
        numbers = [int(line.strip()) for line in f if line.strip().isdigit()]

    # Validate numbers are between 0 and 99
    if not all(0 <= num <= 99 for num in numbers):
        raise ValueError("All numbers must be between 0 and 99")

    # Calculate image dimensions (square image)
    size = int(math.sqrt(len(numbers)))
    if size * size != len(numbers):
        raise ValueError("Number of values must form a perfect square")

    # Create a new grayscale image
    img = Image.new('L', (size, size))  # 'L' mode for grayscale
    pixels = img.load()

    # Populate image pixels with numbers (scaled)
    for i in range(size):
        for j in range(size):
            index = i * size + j
            pixels[j, i] = int(numbers[index] * 255 / 99)  # Scale to 0–255

    # Save the image
    img.save(output_image)
    print(f"Image saved as {output_image}")

# Example usage
if __name__ == "__main__":
    try:
        create_image_from_file("rng_output.txt", "./reports/figures/output.png")
    except Exception as e:
        print(f"Error: {e}")
