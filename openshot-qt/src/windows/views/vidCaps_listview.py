"""
 @file
 @brief This file contains the emojis listview, used by the main window
 @author Jonathan Thomas <jonathan@openshot.org>

 @section LICENSE

 Copyright (c) 2008-2018 OpenShot Studios, LLC
 (http://www.openshotstudios.com). This file is part of
 OpenShot Video Editor (http://www.openshot.org), an open-source project
 dedicated to delivering high quality video editing and animation solutions
 to the world.

 OpenShot Video Editor is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 OpenShot Video Editor is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with OpenShot Library.  If not, see <http://www.gnu.org/licenses/>.
 """


from PyQt5.QtCore import QMimeData, QSize, QPoint, Qt, pyqtSlot, QRegExp, QObjectCleanupHandler, QRectF
from PyQt5.QtGui import QDrag, QPainter, QColor, QImage, QPen, QFont, QPixmap
from PyQt5.QtWidgets import QListView, QCheckBox, QGridLayout, QListWidget, QBoxLayout, QWidget, QListWidgetItem

import openshot  # Python module for libopenshot (required video editing module installed separately)
from classes.query import File
from classes.app import get_app
from classes.logger import log
from classes.time_parts import secondsToTimecode
import json

class VidCapItem(QWidget):
    def __init__(self, checkbox):
        QWidget.__init__(self, flags=Qt.Widget)
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        layout.addWidget(checkbox)
        layout.setSizeConstraint(QBoxLayout.SetFixedSize)
        self.setLayout(layout)

class VidCapsListView(QListWidget):
    """ A QListView QWidget used on the main window """

    def get_bbox(self):
        if not hasattr(self.win, "preview_thread"):
            return "00:00:00.000", []
        
        # get current timestamp
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])
        current_position = int((self.win.preview_thread.current_frame - 1) / fps_float)
        current_timestamp = secondsToTimecode(current_position, fps["num"], fps["den"], use_milliseconds=True)

        
        # get detection result with current timestamp
        current_bbox = []
        if current_position in self.win.timeline_sync.detections:
            current_bbox = self.win.timeline_sync.detections[current_position]
        return current_position, current_bbox

    def update(self):
        timestamp, self.bboxs = self.get_bbox()
        # Refresh Layout
        cleaner = QObjectCleanupHandler()
        cleaner.add(self.layout())
        cleaner.clear()

        self.listCheckBox = self.bboxs[::]

        layout = QBoxLayout(QBoxLayout.TopToBottom)
        self.viewer = QListWidget(self)

        for i, v in enumerate(self.listCheckBox[1:-1]):
            if type(v) == int:
                continue
            # Make checkbox
            self.listCheckBox[i] = QCheckBox(v["translation"], self)
            # Save check status in bboxs
            if 'checked' not in self.bboxs[i+1]:
                self.bboxs[i+1]['checked'] = True
            self.listCheckBox[i].setChecked(self.bboxs[i+1]['checked']) 
            # Connect checkbox ChangeState function
            self.listCheckBox[i].stateChanged.connect(self.clickBoxStateChanged(self.bboxs[i+1]))
            # Add checkbox to Widget
            item = QListWidgetItem(self.viewer)
            custom_widget = VidCapItem(self.listCheckBox[i])
            item.setSizeHint(custom_widget.sizeHint())
            self.viewer.setItemWidget(item, custom_widget)
            self.viewer.addItem(item)

        layout.addWidget(self.viewer)
        self.setLayout(layout)

#ing modify
    def makeTextInBbox(self, bbox) :
        # set bbox bg_color and font color
        bg_color = QColor(Qt.black)
        font_color = QColor(Qt.white)
        title_string = ""

        image = QPixmap(0, 0)

        qp = QPainter()
        qp.begin(self)

        qp.setBrush(bg_color)
        qp.drawRect(0, 0,bbox['point'][2],bbox['point'][3])

        pen = QPen(font_color)
        pen.setWidth(2)
        qp.setPen(pen)

        font = QFont()
        font.setFamily('Times')
        font.setBold(True)
        font.setPointSize(24)
        qp.setFont(font)

        qp.drawText(150, 250, "bbox['translation']")
        label1 = QLabel(self)
        
        #이미지 관련 클래스와 라벨 연결 
        label1.setPixmap(image)
 
        self.show()

        qp.end()
        
    def clickBoxStateChanged(self, bbox):
        def stateChanged(state):
            if state == 0:
                bbox['checked'] = False
            else:
                bbox['checked'] = True
                # self.makeTextInBbox(bbox)
            print(bbox)
            self.win.videoPreview.repaint()

            # [later] repaint bbox
        return stateChanged

    def __init__(self):
        # Invoke parent init
        QListView.__init__(self)

        # Get external references
        app = get_app()
        _ = app._tr
        self.win = app.window   

        # Setup View style
        self.setViewMode(QListView.ListMode)
        self.setResizeMode(QListView.Adjust)
        self.setStyleSheet('QListView::item { padding-top: 2px; }') 
            