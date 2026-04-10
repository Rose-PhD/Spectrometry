import os
import pathlib
import matplotlib.pyplot as plt

join_path = lambda parent_pth, child_pth: os.path.join(parent_pth, child_pth)

def load_data(data_path, spectral_buffer, image_buffer):
    for week in os.listdir(data_path):
        week_pth = join_path(data_path, week)
        for _class in os.listdir(week_pth):
            reading_pth = join_path(week_pth, _class)
            for plant in os.listdir(reading_pth):
                plant_path = join_path(reading_pth, plant)
                for label in os.listdir(plant_path):
                    label_pth = join_path(plant_path, label)
                    for specimen_dir in os.listdir(label_pth):
                        specimen_path = join_path(label_pth, specimen_dir)
                        for specimen_file in os.listdir(specimen_path):
                            specimen_file_path = join_path(specimen_path, specimen_file)
                            if specimen_file.endswith('csv'):
                                spectral_buffer.append(specimen_file_path)
                            elif specimen_file.endswith('.jpg'):
                                image_buffer.append(specimen_file_path)
                            else:
                                raise ValueError('Unsupported file type detected')



def plot_spectral(_range, values):
    plt.plot(_range, values)
    plt.title('Spectral characteristic')
    plt.xlabel('Wavelength (400-700nm)')
    plt.ylabel('Normalized Intensity')
    plt.tight_layout()
    plt.show()

display_img = lambda img_path: plt.imshow(mpimg.imread(img_path))