import sys
import os

def check_path(loc):
    if sys.platform == "linux" or sys.platform == "linux2":
        locLinux = loc.replace("\\", "/")
        return locLinux
    else:
        return loc

def write_object(f, xmin, ymin, xmax, ymax, name):
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


def write_xml(xml_fname, bboxes):
    f = open(xml_fname, 'w')
    height = 281
    width = 281

    line = '<annotation>\n'
    f.writelines(line)

    line = '\t<folder>JPEGImages</folder>\n'
    f.writelines(line)

    line = '\t<filename>' + 'xxx.jpg' + '</filename>\n'
    f.writelines(line)

    path = os.getcwd()
    line = '\t<path>' + os.path.join(path, 'xxx.jpg') + '</path>\n'
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
    for bbox in bboxes:
        [ymin, xmin, ymax, xmax, name] = bbox
        write_object(f, str(xmin), str(ymin), str(xmax), str(ymax), str(name))

    line = '</annotation>\n'
    f.writelines(line)

    f.close()

