require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

width = 240;
height = 240;
noFrames = 5;

collectgarbage("collect")
print '==> downloading dataset'
print(collectgarbage("count") * 1024)
platform = 'linux'
--[[
Note: The dataset can be obtained upon request for limited use only. Please feel free to drop an email at qureshi.ahmed@irl.sys.es.osaka-u.ac.jp with the subject "Gesture Motion Dataset".
--]]
if platform == 'linux' then
  dir_general = 'data/processedFramesAug/' --- dataset location 

end

classPath = {'1_konichiwa', '2_congrats', '3_hello', '4_goodbye', '5_sorry', '6_dontKnow', '7_shh', '8_hi5', '9_standing'}
gestureNo = 9     
gesturePath ={}

for i= 1, 9 do
    gesturePath[i] = dir_general .. classPath[i]..'/'
end


function scandir(director)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls -a "'..director..'"'):lines() do
        i = i + 1
        t[i] = filename
    end
    return t
end

function imageLoader(director, size)
  local arrayDir = scandir(director)
  local images = torch.Tensor(noFrames, width, height)
    local imageRGB = torch.Tensor(3, width, height)
  imCounter = 1;
  for ind =1,#arrayDir do
    if type(string.find(arrayDir[ind], '.jpg'))~='nil' then
      if imCounter > size then break end
        imageRGB = image.load(director .. '/' .. arrayDir[ind])
            
        images[{{imCounter}}] = image.rgb2y(imageRGB)
        imCounter = imCounter +1
    end
  end
  return images
end

string.split_it = function(str, sep)
   if str == nil then return nil end
   return string.gmatch(str, "[^\\" .. sep .. "]+")
end
string.split = function(str, sep)
   local ret = {}
   for seg in string.split_it(str, sep) do
      ret[#ret+1] = seg
   end
   return ret
end


dataSize = 0;
for i = 1,9 do 
  gestureFolderList = scandir(gesturePath[i])
    print(gesturePath[i])
    print('size' .. #gestureFolderList-3)
  dataSize = dataSize + #gestureFolderList-3
end
--double dataSize for augm
dataSize = dataSize *scale 
print('dataSize is' .. dataSize)
allImages = torch.Tensor(dataSize, noFrames, desiredWidth, desiredHeight)
allLabels = torch.Tensor(dataSize)
videoAnnotationHuman = {}
videoAnnotationPlace = {}
--#gestureFolderList

count = 1               --loads all the images per class, and annotates them, connection is count 
for i = 1,9 do 
  gestureFolderList = scandir(gesturePath[i])
    --print(#gestureFolderList)
    for j = 4,#gestureFolderList do
      local dir = gesturePath[i] .. gestureFolderList[j] 
    --print(dir) 
        
      local initialIm = torch.Tensor(5, height, width)
      initialIm =   imageLoader(dir, noFrames)
      allImages[{{count, count +1}, {1,noFrames}, {}, {}}] = data_augmentation(initialIm)
      stringVideo = string.split(gestureFolderList[j], '_')
      videoAnnotationHuman[count] = stringVideo[1]      --participant
      videoAnnotationHuman[count+1] = stringVideo[1] 
      videoAnnotationPlace[count] = stringVideo[2]      --place
      videoAnnotationPlace[count+1] = stringVideo[2] 
      allLabels[count]=i 
      allLabels[count+1]=i 
      count = count +2 --scale
     end
 end


function table_count(tt, item)
  local count
  count = 0
  for ii,xx in pairs(tt) do
    if item == xx then count = count + 1 end
  end
  return count
end

print '==> preprocessing data: floating, shuffling and normalizing'


labelsShuffle = torch.randperm((#allLabels)[1])  
tesize = table_count(videoAnnotationPlace,testNewLocation)

trsize=dataSize - tesize 

trainData = {
   data= torch.Tensor(trsize, noFrames, desiredWidth, desiredHeight),    
   labels = torch.Tensor(trsize),
   positionTensor = torch.Tensor(trsize),  --to make the mapping 
   size = function() return trsize end
}

testData ={
   data = torch.Tensor(tesize, noFrames, desiredWidth, desiredHeight),   
   labels = torch.Tensor(tesize),
   positionTensor = torch.Tensor(tesize),
   size = function() return tesize end
}
trainCount = 1
testCount = 1
for i=1,dataSize do             
  if videoAnnotationPlace[labelsShuffle[i]]~=testNewLocation then
        
    trainData.data[trainCount] = allImages[labelsShuffle[i]]:clone()
    
    trainData.labels[trainCount] = allLabels[labelsShuffle[i]]
    trainData.positionTensor[trainCount] = i
    --itorch.image(trainData.data[trainCount][1])
    trainCount = trainCount +1
        
   
  else 
    testData.data[testCount] = allImages[labelsShuffle[i]]:clone()
    
    testData.labels[testCount] = allLabels[labelsShuffle[i]]
    testData.positionTensor[testCount] = i
    testCount = testCount +1
end
end
print(' The size of the testData ' .. testNewLocation .. testData.size())

trainData.data = trainData.data:cuda()
testData.data = testData.data:cuda()


-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
frames={1,2,3,4,5}

   mean[1] = trainData.data[{ {},{},{},{} }]:mean()
   std[1] = trainData.data[{ {},{},{},{} }]:std()
   trainData.data[{ {},{},{},{} }]:add(-mean[1])
   trainData.data[{ {},{},{},{} }]:div(std[1])
   --save it to file 
   dataChar = io.open('MeadSTD', 'a')
   dataChar:write(' Location '.. testNewLocation..  ' mean  '.. mean[1] .. ' std  '.. std[1] )
   dataChar:close()
-- Normalize test data, using the training means/stds

   testData.data[{ {},{},{},{} }]:add(-mean[1])
   testData.data[{ {},{},{},{} }]:div(std[1])

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.


   trainMean = trainData.data[{ {} }]:mean()
   trainStd = trainData.data[{ {} }]:std()

   testMean = testData.data[{ {}}]:mean()
   testStd = testData.data[{ {} }]:std()

   print('training data,  mean: ' .. trainMean)
   print('training data, standard deviation: ' .. trainStd)

   print('test data, mean: ' .. testMean)
   print('test data, standard deviation: ' .. testStd)


----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().


 if itorch then
  for i = 1,trsize do
    itorch.image(trainData.data[i])
    print(videoAnnotationPlace[labelsShuffle[trainData.positionTensor[i]] ])
  print(videoAnnotationHuman[labelsShuffle[trainData.positionTensor[i]]])
   print(trainData.labels[i])
end
  

else
  print("For visualization, run this script in an itorch notebook")
end
