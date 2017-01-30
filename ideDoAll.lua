----------------------------------------------------------------------
-- This script an the the called files are based on the tutorial by Clement Farabet: http://code.madbits.com/wiki/doku.php?id=tutorial_supervised
-- The code is modified to allow for crossvalidation, the data is augmented and the parameters are fine tuned 
-- for our task: gesture classification.
--  
 
--------------------------------------------------------------------
require 'torch'

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
-- model:
cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 5, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')         --large video classification
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'cuda', 'type: double | float | cuda')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all in for loop for crossV'

locations = {'buildingEntrance', 'coffeeOutdoor', 'labDoor', 'office', 'corridor', 'buildingOutdoor'}

----------------------------------------------------------------------
print '==> training!'
-- initialize
noOfEpochs = 161


for j=1,#locations do

   testNewLocation = locations[j]
   print(locations[j])

   meanTrainError = 0
   meanValError = 0
   epoch = 1
   opt.save= 'results'..testNewLocation
   dofile 'ideAugm.lua'
   dofile 'ideData.lua'
   dofile 'ideModel.lua'
   dofile 'ideLoss.lua'
   dofile 'ideTrain.lua'
   dofile 'ideTest.lua'
   local meanTrainErrorLog = torch.FloatTensor(noOfEpochs)
   local meanValErrorLog = torch.FloatTensor(noOfEpochs)
   

   for i=1,noOfEpochs do
      previousValError = meanValError
      previousTrainError = meanTrainError

      train()
      test()

      print('meanValError' ..meanValError)
      print('meanTrainError' ..meanTrainError)
      if (i % 20 == 0 or i ==1) then
        	   	-- save/log current net
   	   local filename = paths.concat(opt.save, 'model' .. i ..'.net')
   	   os.execute('mkdir -p ' .. sys.dirname(filename))
   	   print('==> saving model to '..filename)
   	   torch.save(filename, model)
   	end
   --if (epoch > 10 )
      if meanTrainError < previousTrainError and  meanValError > previousValError then
         print('WARNING! Overfitting might occur')
         print('lastValidation' .. previousValError .. 'currentValidation' ..meanValError)
         print('lastTRaining' .. previousTrainError .. 'currentTraining' .. meanTrainError)
         if meanTrainError< meanValError + 0.3 then
            print('STOP HERE')
	end
      end
      meanValErrorLog[i] =   meanValError
      meanTrainErrorLog[i] = meanTrainError 

   end

   filename2 = paths.concat(opt.save, "meanTrainErrorLog.txt")
   torch.save(filename2,meanTrainErrorLog, ASCII)
   filename3 = paths.concat(opt.save, "meanValErrorLog.txt")
   torch.save(filename3,meanValErrorLog, ASCII)


end
