%%getting image from IP webcam
url = 'http://192.168.43.1:8080/shot.jpg';
I  = imread(url);

%% face detection 
faceDetector = vision.CascadeObjectDetector;
faceDetector.MinSize=[90 90];
NotYet = false;
while ~NotYet
    pause(2);
    I = imread(url);
    bboxes = step(faceDetector, I);
    if ~isempty(bboxes)
        NotYet = true;
    disp('face found!');
    break;
    end
    disp('no face detected :(, repeating...');
end

IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, 'Face');
figure, imshow(IFaces), title('Detected faces');
%% loading classifier
load 'resNet_crop.mat'
%% classification
sizeBox=size(bboxes)
numface=sizeBox(1,1)
for i=1:numface
    faceImage=imcrop(I,bboxes(i,:));
    faceImage=imresize(faceImage,[224 224]);
    label=classify(resNet_crop,faceImage);
    I = insertObjectAnnotation(I, 'rectangle', bboxes(i,:),char(label), 'FontSize', 35);
end
    
figure,imshow(I);
