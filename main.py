import os
from lib.ops import check_path
from write_crops import write_crops

if __name__ == '__main__':
    crop_size = (300, 300)
    save_folder = os.path.join(os.getcwd(), 'data', 'LiVPa')

    # input images
    input_images = []
    input_images.append(check_path(
        'D:\\Jahandar\\Lab\\images\\crops_for_badri_proposal\\LiVPa\\ARBc_#4_Li+VPA_37C_4110_C10_IlluminationCorrected_stitched.tif'))
    input_images.append(check_path(
        'D:\\Jahandar\\Lab\\images\\crops_for_badri_proposal\\LiVPa\\ARBc_#4_Li+VPA_37C_4110_C7_IlluminationCorrected_stitched.tif'))

    # input centers
    centers_fname = check_path('D:\\Jahandar\\Lab\\images\\crops_for_badri_proposal\\LiVPa\\centers.txt')

    # generate crops and xmls
    write_crops(input_images, centers_fname, crop_size=crop_size, adjust_hist=False)

    # generate tfrecord from xmls
    # use generate_tfrecord.py file

    # train