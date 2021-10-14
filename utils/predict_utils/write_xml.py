from xml.dom.minidom import Document
import os


def write_xml(save_path, folder, filename, path, size, object_list):
    doc = Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)
    folders = doc.createElement('folder')
    folder_text = doc.createTextNode(folder)
    folders.appendChild(folder_text)
    root.appendChild(folders)

    filenames = doc.createElement('filename')
    filename_text = doc.createTextNode(filename)
    filenames.appendChild(filename_text)
    root.appendChild(filenames)

    paths = doc.createElement('path')
    path_text = doc.createTextNode(path)
    paths.appendChild(path_text)
    root.appendChild(paths)

    sources = doc.createElement('source')
    root.appendChild(sources)
    database_ = doc.createElement('database')
    database_text = doc.createTextNode('Unknown')
    database_.appendChild(database_text)
    sources.appendChild(database_)

    sizes = doc.createElement('size')
    root.appendChild(sizes)
    widths = doc.createElement('width')
    widths_text = doc.createTextNode(size['width'])
    widths.appendChild(widths_text)
    sizes.appendChild(widths)

    segmented_ = doc.createElement('segmented')
    segmented_text = doc.createTextNode('0')
    segmented_.appendChild(segmented_text)
    root.appendChild(segmented_)

    heights = doc.createElement('height')
    height_text = doc.createTextNode(size['height'])
    heights.appendChild(height_text)
    sizes.appendChild(heights)

    depths = doc.createElement('depth')
    depth_text = doc.createTextNode(size['depth'])
    depths.appendChild(depth_text)
    sizes.appendChild(depths)

    for objects in object_list:
        object_ = doc.createElement('object')
        root.appendChild(object_)
        names = doc.createElement('name')
        name_text = doc.createTextNode(objects['name'])
        names.appendChild(name_text)
        object_.appendChild(names)

        poses = doc.createElement('pose')
        poses_txt = doc.createTextNode('Unspecified')
        poses.appendChild(poses_txt)
        object_.appendChild(poses)

        truncated_ = doc.createElement('truncated')
        truncated_text = doc.createTextNode('0')
        truncated_.appendChild(truncated_text)
        object_.appendChild(truncated_)

        difficult_ = doc.createElement('difficult')
        difficult_text = doc.createTextNode('0')
        difficult_.appendChild(difficult_text)
        object_.appendChild(difficult_)

        bandbox = doc.createElement('bndbox')
        object_.appendChild(bandbox)
        xmins = doc.createElement('xmin')
        xmin_text = doc.createTextNode(objects['bandbox']['xmin'])
        xmins.appendChild(xmin_text)
        bandbox.appendChild(xmins)

        ymins = doc.createElement('ymin')
        ymin_text = doc.createTextNode(objects['bandbox']['ymin'])
        ymins.appendChild(ymin_text)
        bandbox.appendChild(ymins)

        xmaxs = doc.createElement('xmax')
        xmax_text = doc.createTextNode(objects['bandbox']['xmax'])
        xmaxs.appendChild(xmax_text)
        bandbox.appendChild(xmaxs)

        ymaxs = doc.createElement('ymax')
        ymax_text = doc.createTextNode(objects['bandbox']['ymax'])
        ymaxs.appendChild(ymax_text)
        bandbox.appendChild(ymaxs)

        scores = doc.createElement('scores')
        scores_text = doc.createTextNode(objects['scores'])
        scores.appendChild(scores_text)
        object_.appendChild(scores)

    f = open(save_path, 'w')
    doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
    f.close()


if __name__ == '__main__':
    size = dict(width='100',
                height='100',
                depth='3')
    object_ = dict(name='name',
                   bandbox=dict(xmin='1',
                                ymin='2',
                                xmax='3',
                                ymax='4'
                                ),
                   scores='100'
                   )
    objects = [object_, object_]
    if not os.path.exists('./xml_result'):
        os.mkdir('./xml_result')
    save_path = './xml_result/test.xml'
    write_xml(save_path, './', 'a.txt', 'path', size, objects)
