from PIL import Image
import scipy.io as scio
from pylab import *
path = r'E:\CWRU\dataset'    # Please put the data you need to use in the E:\CWRU\dataset folder
filenames = os.listdir(path)
for item in filenames:
    file_path = os.path.join(path, item)
    file = scio.loadmat(file_path)
    for key in file.keys():
        if 'DE' in key:           # Only DE end
            X = file[key]
            for i in range(200):
                length = 4096
                all_lenght = len(X)  # Total length of data on the DE end
                random_start = np.random.randint(low=0, high=(all_lenght - 2*length))
                sample = X[random_start:random_start + length]   # 4096 consecutive points were randomly selected
                sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))  # Normalization
                sample = np.round(sample*255.0)  # The gray pixel value is obtained
                sample = sample.reshape(64,64) # Convert it to a 64 by 64 image
                im = Image.fromarray(sample)
                im.convert('L').save('E:\\CWRU\\gray_image\\'+str(key)+str(i)+'.jpg',format = 'jpeg')