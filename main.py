import SimpleTransformer.train as st_run
import SimpleTransformer.data_create as dc_run
import SimpleNeuralNetwork.train as sn_run
import ConstantVelocityPlusNN.data_create as cvnn_dc_run
import ConstantVelocityPlusNN.train as cvnn_run
import LSTMWeightBasedLoss.data_create as lswbdc_run 
import LSTMWeightBasedLoss.train as lswbnn_run 
import LSTMWeightBasedLoss.train3 as lswbnn_run3
import NewTransformer.data_create as transformerdc_run
import NewTransformer.train1 as transformernn_run
import NewTransformer.ensemble as ensemble

# sn_run.main()
# st_run.main()
# dc_run.main()
# cvnn_dc_run.main()
# cvnn_run.main()
lswbdc_run.main()
# lswbnn_run.main()
# transformerdc_run.main()
# transformernn_run.main()
#ensemble.main()

lswbnn_run3.main()
