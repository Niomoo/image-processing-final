# Creating files train.txt and test.txt
# for training in Darknet framework
#
# Algorithm:
# Setting up full paths --> List of paths -->
# --> Extracting 15% of paths to save into test.txt file -->
# --> Writing paths into train and test txt files
#
# Result:
# Files train.txt and test.txt with full paths to images


# Importing needed library
import os


"""
Start of:
Setting up full path to directory with labelled images
"""

# Full or absolute path to the folder with images
# Find it with Py file getting-full-path.py
# Pay attention! If you're using Windows, yours path might looks like:
# r'C:\Users\my_name\Downloads\video-to-annotate'
# or:
# 'C:\\Users\\my_name\\Downloads\\video-to-annotate'
full_path_to_images = 'val'

"""
End of:
Setting up full path to directory with labelled images
"""


"""
Start of:
Getting list of full paths to labelled images
"""

# Check point
# Getting the current directory
# print(os.getcwd())

# Changing the current directory
# to one with images
os.chdir(full_path_to_images)

# Check point
# Getting the current directory
# print(os.getcwd())

# Defining list to write paths in
p_test = []


for current_dir, dirs, files in os.walk('.'):
    # Going through all files
    for f in files:
        # Checking if filename ends with '.jpeg'
        if f.endswith('.png'):
            # Preparing path to save into train.txt file
            # Pay attention!
            # If you're using Windows, it might need to change
            # this: + '/' +
            # to this: + '\' +
            # or to this: + '\\' +
            path_to_save_into_txt_files = 'val/' + f

            # Appending the line into the list
            # We use here '\n' to move to the next line
            # when writing lines into txt files
            p_test.append(path_to_save_into_txt_files + '\n')

# Slicing first 15% of elements from the list
# to write into the test.txt file
# p_test = p[:int(len(p) * 0.15)]

# Deleting from initial list first 15% of elements
# p = p[int(len(p) * 0.15):]

"""
End of:
Getting list of full paths to labelled images
"""


"""
Start of:
Creating test.txt files
"""

# Creating file test.txt and writing 15% of lines in it
with open('test.txt', 'w') as test_txt:
    # Going through all elements of the list
    for e in p_test:
        # Writing current path at the end of the file
        test_txt.write(e)

"""
End of:
Creating test.txt files
"""