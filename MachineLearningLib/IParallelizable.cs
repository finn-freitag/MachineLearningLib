﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib
{
    public interface IParallelizable
    {
        IParallelizer Parallelizer { get; set; }
    }
}
