	o???qN@o???qN@!o???qN@	?삼????삼???!?삼???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$o???qN@?H?}8??A?? ?9N@Y?&?W??*	?????ii@2F
Iterator::ModelI??&??!y??? ?F@)??y?):??1??OՂA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?X?? ??!?+???<@)?a??4???1??EGj?7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatem???{???!|?8H0@)=?U?????1???9t?'@:Preprocessing2U
Iterator::Model::ParallelMapV2Q?|a2??!s=??.]$@)Q?|a2??1s=??.]$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ܵ???!?$?eK@)?ZӼ???1??b?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ׁsF??!?$???z@)??ׁsF??1?$???z@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;?O??n??!??	?3?@);?O??n??1??	?3?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?N@aã?!????2@)?I+?v?1wa????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?삼???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?H?}8???H?}8??!?H?}8??      ??!       "      ??!       *      ??!       2	?? ?9N@?? ?9N@!?? ?9N@:      ??!       B      ??!       J	?&?W???&?W??!?&?W??R      ??!       Z	?&?W???&?W??!?&?W??JCPU_ONLYY?삼???b 