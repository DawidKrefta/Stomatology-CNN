from PIL import Image, ImageOps
directory_path_with_treatment = 'C:/Users/dawid/OneDrive/Pulpit/with_treatment/x ('
directory_path_without_treatment = 'C:/Users/dawid/OneDrive/Pulpit/without_treatment/x ('
directory_path_cropped_with = 'C:/Users/dawid/OneDrive/Pulpit/cropped_with/'
directory_path_cropped_without = 'C:/Users/dawid/OneDrive/Pulpit/cropped_without/'
for k in range(1,812):
    whole_path = directory_path_with_treatment + str(k) + ').bmp'
    img = Image.open(whole_path)
    box = (500, 300, 2340, 1380)
    #box2 = (0, -760, 1840, 1080)
    cropped_image = img.crop(box)
   # cropped_2_image = cropped_image.crop(box2)
    flipped = ImageOps.mirror(cropped_image)
    name_save_cropped = directory_path_cropped_with + str(2 * k - 1) + '.jpg'
    name_save_flipped = directory_path_cropped_with + str(2 * k) + '.jpg'
    cropped_image.save(str(name_save_cropped))
    flipped.save(str(name_save_flipped))
