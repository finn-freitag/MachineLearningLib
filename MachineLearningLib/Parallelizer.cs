using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib
{
    public interface IParallelizer
    {
        void Parallelizer(int from, int to, Action<int> action);
    }
}
