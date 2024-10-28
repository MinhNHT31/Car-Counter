from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

# Load model
model = YOLO('yolov10s.pt')

VIDEO_PATH = 'video/vehicle-counting.mp4'

output_path = 'output_video/output_video.mp4'


corner_annotator = sv.BoxCornerAnnotator(
    color=sv.ColorPalette.DEFAULT, thickness=2, corner_length=15, color_lookup=sv.ColorLookup.CLASS
)

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.DEFAULT, text_color=sv.Color.WHITE, 
    text_scale=0.5, text_thickness=1,
    text_position=sv.Position.TOP_LEFT, color_lookup=sv.ColorLookup.CLASS, border_radius=0
)

POLYGON = np.array([[1252, 787],[2298,803],[5039,2159],[-550,2159]])



tracker = sv.ByteTrack()

# Load frames
frames = sv.get_video_frames_generator(VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)


count_in = set()
count_out = set()

# Define the text and its position
text = "Sample Text"
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
text_color = (255, 255, 255)  # White color for text
text_thickness = 1
text_size, _ = cv2.getTextSize(text, fontFace, fontScale, text_thickness)

fps = video_info.fps
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format

video_writer = cv2.VideoWriter(output_path, fourcc, fps, (640,480))

for frame in frames:
    # resize frames
    frame = sv.resize_image(frame,(640,480))
    # Inference
    results = model.predict(frame)

    

    detections = sv.Detections.from_ultralytics(results[0])
    detections = tracker.update_with_detections(detections)
    
    trackerid = [track_id for track_id in detections.tracker_id]

    labels = [
    f"#{track_id}"
    for track_id in detections.tracker_id
    ]

    annotated_frame = frame.copy()


    annotated_frame = corner_annotator.annotate(
    scene=annotated_frame,
    detections=detections
    )

    annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections = detections,
    labels=labels,
    )

    # Display frame by opencv
    x,y,w,h = results[0].boxes.xywh.T

    # draw in and out line
    # in line
    cv2.line(annotated_frame,(0,annotated_frame.shape[1]//2),(annotated_frame.shape[0]//2+100,annotated_frame.shape[1]//2),(255,0,255),2)
    
    # out line
    cv2.line(annotated_frame,(annotated_frame.shape[0]//2+100,annotated_frame.shape[1]//2),(annotated_frame.shape[1],annotated_frame.shape[1]//2),(0,0,255),2)
    
    text_x1 = (annotated_frame.shape[0]//2+100)//2
    text_y1 = (annotated_frame.shape[1]//2)

    text_x2 = int((annotated_frame.shape[0]//2+100)*1.5)
    text_y2 = (annotated_frame.shape[1]//2)

    cv2.rectangle(annotated_frame, (text_x1,text_y1-20), (text_x1+50,text_y1+10), color = (255,0,255), thickness = cv2.FILLED)
    cv2.rectangle(annotated_frame, (text_x2,text_y2-20), (text_x2+60,text_y2+10), color = (0,0,255), thickness = cv2.FILLED)
    
    for cx,cy,id in zip(x,y,trackerid):

        cx = int(cx.item())
        cy = int(cy.item())

        # counting in and out cars
        if 0< cx < annotated_frame.shape[0]//2+50 and annotated_frame.shape[1]//2-2<cy<annotated_frame.shape[1]//2+2:
            count_in.add(id)
        elif cx > annotated_frame.shape[0]//2+40 and annotated_frame.shape[1]//2-4<cy<annotated_frame.shape[1]//2+4:
            count_out.add(id)
        
        cv2.putText(annotated_frame, f'In: {len(count_in)}', (text_x1, text_y1), fontFace, fontScale, text_color, text_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'out: {len(count_out)}', (text_x2, text_y2), fontFace, fontScale, text_color, text_thickness, cv2.LINE_AA)

        cv2.circle(annotated_frame,(cx,cy),2,(255,0,255), thickness=cv2.FILLED)
    
    
    # save video
    video_writer.write(annotated_frame)

    cv2.imshow('frame', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_writer.release()
cv2.destroyAllWindows()
    
