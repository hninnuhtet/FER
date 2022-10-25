%%creating image datastore
rootFolder=fullfile('database');
imds = imageDatastore(fullfile(rootFolder),'IncludeSubFolder',true,...
    'LabelSource','foldernames');

numClasses=numel(categories(imds.Labels));

% splitting imds to training set and test set
[trainingSet,other]=splitEachLabel(imds,0.7,'randomize');
[validationSet,testSet]=splitEachLabel(other,0.5,'randomize');

%% using resnet18 pretrained model and fine tuning
net=resnet18;
imageSize=net.Layers(1).InputSize;
lgraph = layerGraph(net);
clear net;
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc');
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);

%% Preporcessing image data
imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandRotation',[-5 5]);
augmentedTrainingSet=augmentedImageDatastore(imageSize,trainingSet,...
    'ColorPreprocessing','gray2rgb','dataAugmentation',imageAugmenter);
augmentedValidationSet=augmentedImageDatastore(imageSize,validationSet,...
    'ColorPreprocessing','gray2rgb','dataAugmentation',imageAugmenter);
augmentedTestSet=augmentedImageDatastore(imageSize,testSet,...
    'ColorPreprocessing','gray2rgb');


%% defining training options and training 
opts=trainingOptions('sgdm',...
    'InitialLearnRate',0.01,...
    'ExecutionEnvironment','gpu',...
    'MaxEpochs',12,'MiniBatchSize',64,...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedValidationSet, ...
    'ValidationFrequency',20,...
    'ValidationPatience',3,...
    'Plots','training-progress');
resNet_crop=trainNetwork(augmentedTrainingSet,lgraph,opts);

%% Testing the test set accuracy by the use of trained classifier 
predictedLabels=classify(resNet_crop,augmentedTestSet);
accuracy=mean(predictedLabels==testSet.Labels);
X=['Accuracy is ',num2str(accuracy)];
disp(X)
figure;
plotconfusion(testSet.Labels,predictedLabels)
