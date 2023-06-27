import cv2

configFile = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozenModel = "frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozenModel, configFile)

classLabels = []
fileName = "labels.txt"

with open(fileName, "rt") as filePath:
    classLabels = filePath.read().rstrip('\n').split('\n')

print(len(classLabels))
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

capture = cv2.VideoCapture("assets/power.mp4")

while True:
    ret, frame = capture.read()

    if ret == False:
        break

    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)

    if(len(classIndex) > 0):
        for labelIn, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if labelIn <= 80:
                cv2.rectangle(frame, boxes, (0, 255, 0), 3)
                cv2.putText(frame, classLabels[labelIn - 1], (boxes[0] + 10, boxes[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 255), 2)


    cv2.imshow("Detected", frame)

    if (cv2.waitKey(1) == ord('q')):
        break


capture.release()
cv2.destroyAllWindows()