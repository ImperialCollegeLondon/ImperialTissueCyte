# Reads xml file and returns numpy array of markers

from xml.dom import minidom
import numpy as np

def readmarkerxml(xml_path = None):
    if xml_path is None:
        xml_path = raw_input('XML file path (drag-and-drop): ')

    xml_doc = minidom.parse(xml_path.strip('\'').rstrip())

    marker_x = xml_doc.getElementsByTagName('MarkerX')
    marker_y = xml_doc.getElementsByTagName('MarkerY')
    marker_z = xml_doc.getElementsByTagName('MarkerZ')

    marker = np.empty((0,3), int)

    for elem in range (0, marker_x.length):
        marker = np.vstack((marker, [int(marker_x[elem].firstChild.data), int(marker_y[elem].firstChild.data), int(marker_z[elem].firstChild.data)]))

    return marker
