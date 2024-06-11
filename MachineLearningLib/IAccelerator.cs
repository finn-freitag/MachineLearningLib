using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib
{
    public interface IAccelerator
    {
        float DotProduct(float[] vec1, float[] vec2);
        float DotProductT(float[] vec1, float[][] vec2, int subIndex);
        float[] Add(float[] vec1, float[] vec2);
        float[] Add(float a, float[] vec);
        float[] Multiply(float[] vec1, float[] vec2);
        float[] Multiply(float a, float[] vec);
    }
}
