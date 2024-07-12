import os
import tarfile
import urllib.request

def download_imagenette(root='./data', size='full'):
    """
    Download ImageNette dataset.
    :param root: Root directory to store the dataset
    :param size: 'full' for full-size images, '320px' for 320px images, or '160px' for 160px images
    """
    if size not in ['full', '320px', '160px']:
        raise ValueError("Size must be 'full', '320px', or '160px'")

    url = f'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-{size}.tgz'
    filename = os.path.join(root, f'imagenette2-{size}.tgz')
    extracted_folder = os.path.join(root, f'imagenette2-{size}')

    # Create directory if it doesn't exist
    os.makedirs(root, exist_ok=True)

    # Download the file
    if not os.path.exists(filename):
        print(f"Downloading ImageNette ({size})...")
        urllib.request.urlretrieve(url, filename)
        print("Download completed.")

    # Extract the file
    if not os.path.exists(extracted_folder):
        print("Extracting files...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=root)
        print("Extraction completed.")

    print(f"ImageNette dataset is ready at {extracted_folder}")
    return extracted_folder

# Example usage
dataset_path = download_imagenette(size='160px')
print(f"Dataset path: {dataset_path}")