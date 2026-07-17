import numpy as np
import matplotlib.pyplot as plt

def visualize_spectrogram(file_path, title="Spectrogram", save_path=None):
    """
    Loads a .npy file and displays it as a visual image.
    Optionally saves it as a PNG.
    """
    # Load the raw spectrogram image into a plot
    data = np.load(file_path)
    plt.figure(figsize=(10, 4))

    # Display the image
    # - aspect='auto' makes sure it fits the screen nicely
    # - origin='lower' is CRITICAL: it puts low frequencies at the bottom and high at the top
    # - cmap='magma' or 'viridis' applies beautiful colors to our 0.0-1.0 grayscale data
    plt.imshow(data, aspect='auto', origin='lower', cmap='magma')

    # Add styling
    plt.colorbar(label='Normalized Amplitude')
    plt.title(title)
    plt.ylabel('Frequency Bins')
    plt.xlabel('Time (Frames)')
    plt.tight_layout()

    # Save to a file if requested (great for your project presentation!)
    if save_path:
        plt.savefig(save_path, dpi=300)  # dpi=300 makes it high resolution
        print(f"Saved visual to {save_path}")

    # Show it on your screen
    plt.show()