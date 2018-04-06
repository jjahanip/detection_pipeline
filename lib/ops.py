import sys
import os
import pandas as pd
import xml.etree.ElementTree as ET


def check_path(loc):
    if sys.platform == "linux" or sys.platform == "linux2":
        locLinux = loc.replace("\\", "/")
        return locLinux
    if sys.platform == "win32" or sys.platform == "win64":
        locWin = loc.replace("/", '\\')
        return locWin
    else:
        return loc


def write_object(f, xmin, ymin, xmax, ymax, name):
    '''
    Write object information in xml file
    :param f: writer object
    :param xmin: top left corner x
    :param ymin: top left corner y
    :param xmax: bottom right corner x
    :param ymax: bottom right corner y
    :param name: string name of the ckass
    :return: None
    '''
    line = '\t<object>\n'
    f.writelines(line)

    line = '\t\t<name>' + name +'</name>\n'
    f.writelines(line)

    line = '\t\t<pose>Unspecified</pose>\n'
    f.writelines(line)

    line = '\t\t<truncated>0</truncated>\n'
    f.writelines(line)

    line = '\t\t<difficult>0</difficult>\n'
    f.writelines(line)

    line = '\t\t<bndbox>\n'
    f.writelines(line)

    line = '\t\t\t<xmin>' + xmin + '</xmin>\n'
    f.writelines(line)

    line = '\t\t\t<ymin>' + ymin + '</ymin>\n'
    f.writelines(line)

    line = '\t\t\t<xmax>' + xmax + '</xmax>\n'
    f.writelines(line)

    line = '\t\t\t<ymax>' + ymax + '</ymax>\n'
    f.writelines(line)

    line = '\t\t</bndbox>\n'
    f.writelines(line)

    line = '\t</object>\n'
    f.writelines(line)


def write_xml(xml_fname, bboxes, labels, image_size=(300, 300, 1)):
    '''
    Write xml file for single image
    :param xml_fname: /path/to/xml_filename
    :param bboxes: np.array (x,4) containing [xmin, ymin, width, height] of each bounding box
    :param labels: list (x,) contating x strings of class names
    :param image_size: size of the image (width, height, depth)
    :return: None
    '''
    # get file name and path information
    base_name = os.path.basename(xml_fname)
    fname = os.path.splitext(base_name)[0]
    parent_folder = os.path.dirname(xml_fname)
    parent_parent_folder = os.path.dirname(parent_folder)

    f = open(xml_fname, 'w')
    width, height, depth = image_size

    line = '<annotation>\n'
    f.writelines(line)

    line = '\t<folder>img</folder>\n'
    f.writelines(line)

    line = '\t<filename>' + fname + '.jpeg' + '</filename>\n'
    f.writelines(line)

    line = '\t<path>' + parent_parent_folder + '</path>\n'
    f.writelines(line)

    line = '\t<source>\n'
    f.writelines(line)

    line = '\t\t<database>Unknown</database>\n'
    f.writelines(line)

    line = '\t</source>\n'
    f.writelines(line)

    line = '\t<size>\n'
    f.writelines(line)

    line = '\t\t<width>' + str(height) +'</width>\n'
    f.writelines(line)

    line = '\t\t<height>' + str(width) +'</height>\n'
    f.writelines(line)

    line = '\t\t<depth>3</depth>\n'
    f.writelines(line)

    line = '\t</size>\n'
    f.writelines(line)

    line = '\t<segmented>0</segmented>\n'
    f.writelines(line)

    # write the object information
    for bbox, label in zip(bboxes, labels):
        [xmin, ymin, cell_width, cell_height] = bbox
        xmax = xmin + cell_width
        ymax = ymin + cell_height
        write_object(f, str(xmin), str(ymin), str(xmax), str(ymax), str(label))

    line = '</annotation>\n'
    f.writelines(line)

    f.close()


def xml_to_csv(path):
    '''
    Create cvs file for generating cvs file
    :param path: path/to/the/xmls
    :return: path to the saved file name

    Note: keep images in "imgs" folder and xml files in "xmls" folder in same folder
    '''
    xml_list = []
    for xml_file in os.listdir(path):
        tree = ET.parse(os.path.join(path, xml_file))
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    save_path = os.path.dirname(path)
    xml_df.to_csv(os.path.join(save_path, 'labels.csv'), index=None)

    return os.path.join(save_path, 'labels.csv')
