import joblib
import numpy as np
import pandas as pd
from buaiir_spectra.data_v2.dataset import Device, SpectralDataLoader, SpectralDataset
from pathlib import Path

DATA_PATH = Path("/home/wilfred/Downloads/spectra_data_clean")
device = Device.SCAN_CORDER

ds = SpectralDataset(DATA_PATH, device=device)
dl = SpectralDataLoader(ds, batch_size=4)


def load_split(dl: SpectralDataLoader, device=Device) -> None:
    X, Images, Y = [], [], []

    i = -1

    for i, batch in enumerate(dl):
        print(f"Working with {i+1} batch", end=": ")
        x, img, y = batch
        print(f"x: {x.shape}", f"y:{y.shape}", f"Image: {img.shape}")

        X.append(x)
        Images.append(img)
        Y.append(y)

        if (i + 1) % 2 == 0:
            X_np = np.vstack(X)
            Y_df = pd.concat(Y, axis=0)

            # FIX: Replaced non-existent np.vecmat with np.vstack
            Images_np = np.vstack(Images)

            state = {"X": X_np, "y": Y_df, "images": Images_np}
            print(f"Saving {i+1} batch (x, y, images) data to disk....")
            joblib.dump(
                state,
                f"src/buaiir_spectra/data_splits/{device.name}/{device.name}_{i+1}_batch.pkl",
            )
            print(
                f"Done saving {i+1}_batch, resetting states for next half batch....."
            )

            X.clear()
            Y.clear()
            Images.clear()
            state.clear()

    if X:
        X_np = np.vstack(X)
        Y_df = pd.concat(Y, axis=0)
        Images_np = np.vstack(Images)

        state = {"X": X_np, "y": Y_df, "images": Images_np}
        print(f"Saving {i+1} batch (x, y, images) data to disk....")
        joblib.dump(
            state,
            f"src/buaiir_spectra/data_splits/{device.name}/{device.name}_{i+1}_batch.pkl",
        )
        print("Done saving last remaining batch pieces.")


if __name__ == '__main__':
    print(f"Working with data for {device.name}")
    print("*"*200)
    load_split(dl, device=device)
