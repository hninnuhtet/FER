

load 'resNet_crop.mat'
%%
C = webcamlist
cam=webcam(C{2});
preview(cam);
NotYet = false;
faceDetector = vision.CascadeObjectDetector;
faceDetector.MinSize=[90 90];
while ~NotYet
pause(2);
I = snapshot(cam);
disp('took a snapshot. checking to find a face ....')
bboxes = step(faceDetector, I);
if ~isempty(bboxes)
NotYet = true;
disp('face found!');
break;
end
disp('no face detected :(, repeating...');
end
closePreview(cam);
clear('cam');


IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, 'face');
figure, imshow(IFaces), title('Detected faces');


sizeBox=size(bboxes)
numface=sizeBox(1,1)
for i=1:numface
    faceImage=imcrop(I,bboxes(i,:));
    faceImage=imresize(faceImage,[224 224]);
    label=classify(resNet_crop,faceImage);
    figure,imshow(faceImage);
    title(char(label));
end
