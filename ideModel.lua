require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 9-class problem
noutputs = 9

-- input dimensions
nfeats = 5   --5 frames
width = 210
height = 210
ninputs = nfeats*width*height

-- hidden units, filter sizes (for ConvNet only):
-- CONFIG 1
--nstates = {64,128,128,128,128}
--noNodesLastLayer = 901
nstates = {32,64,82,82}
noNodesLastLayer = 505
filtsize = {5, 10}
poolsize = 2
normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
print '==> construct model'


   
      
      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize[1], filtsize[1], 2, 2,2,2)) --stride of 2 and padding
      --results width of 105
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize))
      --output of 52
     
      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolution( nstates[1],  nstates[2], filtsize[1], filtsize[1],1,1, 2, 2))    --stride of 1 and padding
      --results width of 52 
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize))
      --results width of 26
      

      -- stage 3 :convolution
      model:add(nn.SpatialConvolution( nstates[2],  nstates[3], filtsize[1], filtsize[1],1,1, 1, 1))
      --width of 25
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize))
         --results width of 12
     

      --stage 4: convolution
      model:add(nn.SpatialConvolution( nstates[3],  nstates[4], filtsize[1], filtsize[1],1,1, 1, 1))
      --width of 11
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize))
      --results width of 5

      --stage 6: linear stage 


      -- stage 3 : standard 2-layer neural network
      model:add(nn.View( nstates[4]*5*5))
      model:add(nn.Dropout(0.5))
      --CONFIG1
      --model:add(nn.Linear(nstates[5]*5*5, 901))
      model:add(nn.Linear(nstates[4]*5*5, noNodesLastLayer))
      model:add(nn.ReLU())
      model:add(nn.Linear(noNodesLastLayer, noutputs))
  


----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
	 print '==> visualizing ConvNet filters'
	 print('Layer 1 filters:')
	 itorch.image(model:get(1).weight)
	 print('Layer 2 filters:')
	 itorch.image(model:get(5).weight)
      else
	 print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
