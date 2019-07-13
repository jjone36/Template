
import os

def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst, target_is_directory = True)

dir = '../images/fruits-360/'
new_dir = '../images/fruits_10/'

# Create a new directory
os.mkdir(new_dir)

# The list of fruits to use
fruits = ['Apple Golden 1', 'Apple Red Delicious', 'Avocado', 'Banana', 'Cherry 1',
          'Cocos', 'Kiwi', 'Lemon', 'Limes', 'Mango']

# Designate the paths
train_path = os.path.abspath(dir + 'Training/')
test_path = os.path.abspath(dir + 'Test/')

new_train_path = os.path.abspath(new_dir + 'Training/')
new_test_path = os.path.abspath(new_dir + 'Test/')

# Create new files
os.mkdir(new_train_path)
os.mkdir(new_test_path)

for f in fruits:
    link(train_path + f, new_train_path + f)
    link(test_path + f, new_test_path + f)
