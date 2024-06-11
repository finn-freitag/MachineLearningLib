using MachineLearningLib.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.WeightInitializers
{
    public class XavierWeightInitializer : IWeightInitializer
    {
        int inputcount = 0;
        Random r;

        public XavierWeightInitializer(int seed = -1)
        {
            if(seed == -1)
                r = new Random();
            else
                r = new Random(seed);
        }

        public float GetInitialWeight()
        {
            double variance = 1.0 / Math.Sqrt(inputcount);
            return (float)(r.NextDouble() * 1 * variance - variance);
        }

        public void SetLayer(Layer layer)
        {
            inputcount = layer.PreviousLayer.NeuronsSum.Length;
        }
    }
}
