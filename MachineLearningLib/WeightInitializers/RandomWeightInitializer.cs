using MachineLearningLib.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.WeightInitializers
{
    public class RandomWeightInitializer : IWeightInitializer
    {
        Random r;
        float min;
        float max;

        public RandomWeightInitializer(int seed = -1, float min = -1, float max = 1)
        {
            this.min = min;
            this.max = max;
            if (seed == -1)
                r = new Random();
            else
                r = new Random(seed);
        }

        public float GetInitialWeight()
        {
            return (float)r.NextDouble() * (max - min) + min;
        }

        public void SetLayer(Layer layer)
        {
            
        }
    }
}
