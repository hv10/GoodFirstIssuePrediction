ЊЎ
Ф
8
Const
output"dtype"
valuetensor"
dtypetype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype
Ј
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype

NoOp
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring "serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8нд

index_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_33*
value_dtype0	
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
Х
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*!
fR
__inference_<lambda>_597

NoOpNoOp^PartitionedCall
ѕ
Gindex_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2index_lookup_index_table*
Tkeys0*
Tvalues0	*+
_class!
loc:@index_lookup_index_table*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
	
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*а
valueЦBУ BМ

layer-0
layer_with_weights-0
layer-1
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 

state_variables
	_index_lookup_layer

trainable_variables
	variables
regularization_losses
	keras_api
 
 
 
­
metrics
trainable_variables
	variables
layer_regularization_losses

layers
non_trainable_variables
layer_metrics
regularization_losses
 
 
s
state_variables

_table
trainable_variables
	variables
regularization_losses
	keras_api
 
 
 
­
metrics

trainable_variables
	variables
layer_regularization_losses

layers
non_trainable_variables
layer_metrics
regularization_losses
 
 

0
1
 
 
 
LJ
tableAlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table
 
 
 
­
metrics
trainable_variables
	variables
layer_regularization_losses

 layers
!non_trainable_variables
"layer_metrics
regularization_losses
 
 

	0
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
З
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1index_lookup_index_tableConst*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference_signature_wrapper_403
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameGindex_lookup_index_table_lookup_table_export_values/LookupTableExportV2Iindex_lookup_index_table_lookup_table_export_values/LookupTableExportV2:1Const_1*
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*%
f R
__inference__traced_save_631

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameindex_lookup_index_table*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_restore_646рМ

(
__inference_<lambda>_597
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
ђ]
п
>__inference_model_layer_call_and_return_conditional_losses_501

inputsY
Utext_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_table_handleZ
Vtext_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
identity	ЂHtext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:џџџџџџџџџ2 
text_vectorization/StringLower§
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2'
%text_vectorization/StaticRegexReplaceИ
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
text_vectorization/Squeeze
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2&
$text_vectorization/StringSplit/Constџ
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:2.
,text_vectorization/StringSplit/StringSplitV2Й
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2text_vectorization/StringSplit/strided_slice/stackН
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4text_vectorization/StringSplit/strided_slice/stack_1Н
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4text_vectorization/StringSplit/strided_slice/stack_2д
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask2.
,text_vectorization/StringSplit/strided_sliceЖ
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4text_vectorization/StringSplit/strided_slice_1/stackК
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_1К
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_2­
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask20
.text_vectorization/StringSplit/strided_slice_1ђ
9text_vectorization/StringSplit/RaggedFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2;
9text_vectorization/StringSplit/RaggedFromValueRowIds/Castы
;text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2=
;text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1ї
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ShapeShape=text_vectorization/StringSplit/RaggedFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2E
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Shapeд
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ConstЭ
Btext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ProdProdLtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Shape:output:0Ltext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2D
Btext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Prodд
Gtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater/yй
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/GreaterGreaterKtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Prod:output:0Ptext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2G
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater
Btext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/CastCastItext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2D
Btext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Castи
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_1Н
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MaxMax=text_vectorization/StringSplit/RaggedFromValueRowIds/Cast:y:0Ntext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2C
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MaxЬ
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2E
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add/yЪ
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/addAddV2Jtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Max:output:0Ltext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2C
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/addН
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/mulMulFtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Cast:y:0Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2C
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/mulТ
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MaximumMaximum?text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1:y:0Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2G
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MaximumЦ
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MinimumMinimum?text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1:y:0Itext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2G
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Minimumб
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2G
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_2Є
Ftext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/BincountBincount=text_vectorization/StringSplit/RaggedFromValueRowIds/Cast:y:0Itext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Minimum:z:0Ntext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2H
Ftext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/BincountЦ
@text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum/axisЬ
;text_vectorization/StringSplit/RaggedFromValueRowIds/CumsumCumsumMtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Bincount:bins:0Itext_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2=
;text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsumж
Dtext_vectorization/StringSplit/RaggedFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2F
Dtext_vectorization/StringSplit/RaggedFromValueRowIds/concat/values_0Ц
@text_vectorization/StringSplit/RaggedFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@text_vectorization/StringSplit/RaggedFromValueRowIds/concat/axis
;text_vectorization/StringSplit/RaggedFromValueRowIds/concatConcatV2Mtext_vectorization/StringSplit/RaggedFromValueRowIds/concat/values_0:output:0Atext_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum:out:0Itext_vectorization/StringSplit/RaggedFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ2=
;text_vectorization/StringSplit/RaggedFromValueRowIds/concatю
Htext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Utext_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Vtext_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:2J
Htext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2
1text_vectorization/index_lookup/assert_equal/NoOpNoOp*
_output_shapes
 23
1text_vectorization/index_lookup/assert_equal/NoOpж
(text_vectorization/index_lookup/IdentityIdentityQtext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2*
(text_vectorization/index_lookup/Identityи
*text_vectorization/index_lookup/Identity_1IdentityDtext_vectorization/StringSplit/RaggedFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2,
*text_vectorization/index_lookup/Identity_1Є
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 21
/text_vectorization/RaggedToTensor/default_value
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2)
'text_vectorization/RaggedToTensor/Constћ
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:01text_vectorization/index_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:03text_vectorization/index_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS28
6text_vectorization/RaggedToTensor/RaggedTensorToTensorч
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0I^text_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 2
Htext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2Htext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: 
л
y
#__inference_model_layer_call_fn_392
input_1
unknown
	unknown_0	
identity	ЂStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: 
О
Т
>__inference_model_layer_call_and_return_conditional_losses_385

inputs
text_vectorization_379
text_vectorization_381	
identity	Ђ*text_vectorization/StatefulPartitionedCall 
*text_vectorization/StatefulPartitionedCallStatefulPartitionedCallinputstext_vectorization_379text_vectorization_381*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_text_vectorization_layer_call_and_return_conditional_losses_3292,
*text_vectorization/StatefulPartitionedCallН
IdentityIdentity3text_vectorization/StatefulPartitionedCall:output:0+^text_vectorization/StatefulPartitionedCall*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 2X
*text_vectorization/StatefulPartitionedCall*text_vectorization/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: 
С
У
>__inference_model_layer_call_and_return_conditional_losses_355
input_1
text_vectorization_349
text_vectorization_351	
identity	Ђ*text_vectorization/StatefulPartitionedCallЁ
*text_vectorization/StatefulPartitionedCallStatefulPartitionedCallinput_1text_vectorization_349text_vectorization_351*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_text_vectorization_layer_call_and_return_conditional_losses_3292,
*text_vectorization/StatefulPartitionedCallН
IdentityIdentity3text_vectorization/StatefulPartitionedCall:output:0+^text_vectorization/StatefulPartitionedCall*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 2X
*text_vectorization/StatefulPartitionedCall*text_vectorization/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: 
ђ]
п
>__inference_model_layer_call_and_return_conditional_losses_452

inputsY
Utext_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_table_handleZ
Vtext_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
identity	ЂHtext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:џџџџџџџџџ2 
text_vectorization/StringLower§
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2'
%text_vectorization/StaticRegexReplaceИ
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
text_vectorization/Squeeze
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2&
$text_vectorization/StringSplit/Constџ
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:2.
,text_vectorization/StringSplit/StringSplitV2Й
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2text_vectorization/StringSplit/strided_slice/stackН
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4text_vectorization/StringSplit/strided_slice/stack_1Н
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4text_vectorization/StringSplit/strided_slice/stack_2д
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask2.
,text_vectorization/StringSplit/strided_sliceЖ
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4text_vectorization/StringSplit/strided_slice_1/stackК
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_1К
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_2­
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask20
.text_vectorization/StringSplit/strided_slice_1ђ
9text_vectorization/StringSplit/RaggedFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2;
9text_vectorization/StringSplit/RaggedFromValueRowIds/Castы
;text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2=
;text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1ї
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ShapeShape=text_vectorization/StringSplit/RaggedFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2E
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Shapeд
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ConstЭ
Btext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ProdProdLtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Shape:output:0Ltext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2D
Btext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Prodд
Gtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater/yй
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/GreaterGreaterKtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Prod:output:0Ptext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2G
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater
Btext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/CastCastItext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2D
Btext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Castи
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_1Н
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MaxMax=text_vectorization/StringSplit/RaggedFromValueRowIds/Cast:y:0Ntext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2C
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MaxЬ
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2E
Ctext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add/yЪ
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/addAddV2Jtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Max:output:0Ltext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2C
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/addН
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/mulMulFtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Cast:y:0Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2C
Atext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/mulТ
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MaximumMaximum?text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1:y:0Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2G
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MaximumЦ
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MinimumMinimum?text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1:y:0Itext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2G
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Minimumб
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2G
Etext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_2Є
Ftext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/BincountBincount=text_vectorization/StringSplit/RaggedFromValueRowIds/Cast:y:0Itext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Minimum:z:0Ntext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2H
Ftext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/BincountЦ
@text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum/axisЬ
;text_vectorization/StringSplit/RaggedFromValueRowIds/CumsumCumsumMtext_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Bincount:bins:0Itext_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2=
;text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsumж
Dtext_vectorization/StringSplit/RaggedFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2F
Dtext_vectorization/StringSplit/RaggedFromValueRowIds/concat/values_0Ц
@text_vectorization/StringSplit/RaggedFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@text_vectorization/StringSplit/RaggedFromValueRowIds/concat/axis
;text_vectorization/StringSplit/RaggedFromValueRowIds/concatConcatV2Mtext_vectorization/StringSplit/RaggedFromValueRowIds/concat/values_0:output:0Atext_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum:out:0Itext_vectorization/StringSplit/RaggedFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ2=
;text_vectorization/StringSplit/RaggedFromValueRowIds/concatю
Htext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Utext_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Vtext_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:2J
Htext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2
1text_vectorization/index_lookup/assert_equal/NoOpNoOp*
_output_shapes
 23
1text_vectorization/index_lookup/assert_equal/NoOpж
(text_vectorization/index_lookup/IdentityIdentityQtext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2*
(text_vectorization/index_lookup/Identityи
*text_vectorization/index_lookup/Identity_1IdentityDtext_vectorization/StringSplit/RaggedFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2,
*text_vectorization/index_lookup/Identity_1Є
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 21
/text_vectorization/RaggedToTensor/default_value
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2)
'text_vectorization/RaggedToTensor/Constћ
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:01text_vectorization/index_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:03text_vectorization/index_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS28
6text_vectorization/RaggedToTensor/RaggedTensorToTensorч
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0I^text_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 2
Htext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2Htext_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: 

*
__inference__destroyer_592
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
и
x
#__inference_model_layer_call_fn_510

inputs
unknown
	unknown_0	
identity	ЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: 
О
Т
>__inference_model_layer_call_and_return_conditional_losses_367

inputs
text_vectorization_361
text_vectorization_363	
identity	Ђ*text_vectorization/StatefulPartitionedCall 
*text_vectorization/StatefulPartitionedCallStatefulPartitionedCallinputstext_vectorization_361text_vectorization_363*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_text_vectorization_layer_call_and_return_conditional_losses_3292,
*text_vectorization/StatefulPartitionedCallН
IdentityIdentity3text_vectorization/StatefulPartitionedCall:output:0+^text_vectorization/StatefulPartitionedCall*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 2X
*text_vectorization/StatefulPartitionedCall*text_vectorization/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: 
м
F
__inference__creator_582
identityЂindex_lookup_index_tableЁ
index_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_33*
value_dtype0	2
index_lookup_index_table
IdentityIdentity'index_lookup_index_table:table_handle:0^index_lookup_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 24
index_lookup_index_tableindex_lookup_index_table
л
y
#__inference_model_layer_call_fn_374
input_1
unknown
	unknown_0	
identity	ЂStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: 

,
__inference__initializer_587
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
ѓ

0__inference_text_vectorization_layer_call_fn_577

inputs
unknown
	unknown_0	
identity	ЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_text_vectorization_layer_call_and_return_conditional_losses_3292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: 
Й
w
!__inference_signature_wrapper_403
input_1
unknown
	unknown_0	
identity	ЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__wrapped_model_2762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: 
и
x
#__inference_model_layer_call_fn_519

inputs
unknown
	unknown_0	
identity	ЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: 
КI
Г
K__inference_text_vectorization_layer_call_and_return_conditional_losses_568

inputsF
Bindex_lookup_none_lookup_table_find_lookuptablefindv2_table_handleG
Cindex_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
identity	Ђ5index_lookup/None_lookup_table_find/LookupTableFindV2Z
StringLowerStringLowerinputs*'
_output_shapes
:џџџџџџџџџ2
StringLowerФ
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*'
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2
StaticRegexReplace
SqueezeSqueezeStaticRegexReplace:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2	
Squeezeg
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
StringSplit/ConstГ
StringSplit/StringSplitV2StringSplitV2Squeeze:output:0StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:2
StringSplit/StringSplitV2
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
StringSplit/strided_slice/stack
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!StringSplit/strided_slice/stack_1
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!StringSplit/strided_slice/stack_2т
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask2
StringSplit/strided_slice
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!StringSplit/strided_slice_1/stack
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#StringSplit/strided_slice_1/stack_1
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#StringSplit/strided_slice_1/stack_2Л
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
StringSplit/strided_slice_1Й
&StringSplit/RaggedFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2(
&StringSplit/RaggedFromValueRowIds/CastВ
(StringSplit/RaggedFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2*
(StringSplit/RaggedFromValueRowIds/Cast_1О
0StringSplit/RaggedFromValueRowIds/bincount/ShapeShape*StringSplit/RaggedFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:22
0StringSplit/RaggedFromValueRowIds/bincount/ShapeЎ
0StringSplit/RaggedFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0StringSplit/RaggedFromValueRowIds/bincount/Const
/StringSplit/RaggedFromValueRowIds/bincount/ProdProd9StringSplit/RaggedFromValueRowIds/bincount/Shape:output:09StringSplit/RaggedFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 21
/StringSplit/RaggedFromValueRowIds/bincount/ProdЎ
4StringSplit/RaggedFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 26
4StringSplit/RaggedFromValueRowIds/bincount/Greater/y
2StringSplit/RaggedFromValueRowIds/bincount/GreaterGreater8StringSplit/RaggedFromValueRowIds/bincount/Prod:output:0=StringSplit/RaggedFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 24
2StringSplit/RaggedFromValueRowIds/bincount/Greaterв
/StringSplit/RaggedFromValueRowIds/bincount/CastCast6StringSplit/RaggedFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 21
/StringSplit/RaggedFromValueRowIds/bincount/CastВ
2StringSplit/RaggedFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2StringSplit/RaggedFromValueRowIds/bincount/Const_1ё
.StringSplit/RaggedFromValueRowIds/bincount/MaxMax*StringSplit/RaggedFromValueRowIds/Cast:y:0;StringSplit/RaggedFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 20
.StringSplit/RaggedFromValueRowIds/bincount/MaxІ
0StringSplit/RaggedFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :22
0StringSplit/RaggedFromValueRowIds/bincount/add/yў
.StringSplit/RaggedFromValueRowIds/bincount/addAddV27StringSplit/RaggedFromValueRowIds/bincount/Max:output:09StringSplit/RaggedFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 20
.StringSplit/RaggedFromValueRowIds/bincount/addё
.StringSplit/RaggedFromValueRowIds/bincount/mulMul3StringSplit/RaggedFromValueRowIds/bincount/Cast:y:02StringSplit/RaggedFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 20
.StringSplit/RaggedFromValueRowIds/bincount/mulі
2StringSplit/RaggedFromValueRowIds/bincount/MaximumMaximum,StringSplit/RaggedFromValueRowIds/Cast_1:y:02StringSplit/RaggedFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 24
2StringSplit/RaggedFromValueRowIds/bincount/Maximumњ
2StringSplit/RaggedFromValueRowIds/bincount/MinimumMinimum,StringSplit/RaggedFromValueRowIds/Cast_1:y:06StringSplit/RaggedFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 24
2StringSplit/RaggedFromValueRowIds/bincount/MinimumЋ
2StringSplit/RaggedFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 24
2StringSplit/RaggedFromValueRowIds/bincount/Const_2Х
3StringSplit/RaggedFromValueRowIds/bincount/BincountBincount*StringSplit/RaggedFromValueRowIds/Cast:y:06StringSplit/RaggedFromValueRowIds/bincount/Minimum:z:0;StringSplit/RaggedFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ25
3StringSplit/RaggedFromValueRowIds/bincount/Bincount 
-StringSplit/RaggedFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-StringSplit/RaggedFromValueRowIds/Cumsum/axis
(StringSplit/RaggedFromValueRowIds/CumsumCumsum:StringSplit/RaggedFromValueRowIds/bincount/Bincount:bins:06StringSplit/RaggedFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2*
(StringSplit/RaggedFromValueRowIds/CumsumА
1StringSplit/RaggedFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 23
1StringSplit/RaggedFromValueRowIds/concat/values_0 
-StringSplit/RaggedFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-StringSplit/RaggedFromValueRowIds/concat/axisЛ
(StringSplit/RaggedFromValueRowIds/concatConcatV2:StringSplit/RaggedFromValueRowIds/concat/values_0:output:0.StringSplit/RaggedFromValueRowIds/Cumsum:out:06StringSplit/RaggedFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ2*
(StringSplit/RaggedFromValueRowIds/concat
5index_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Bindex_lookup_none_lookup_table_find_lookuptablefindv2_table_handle"StringSplit/StringSplitV2:values:0Cindex_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:27
5index_lookup/None_lookup_table_find/LookupTableFindV2^
index_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2 
index_lookup/assert_equal/NoOp
index_lookup/IdentityIdentity>index_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2
index_lookup/Identity
index_lookup/Identity_1Identity1StringSplit/RaggedFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2
index_lookup/Identity_1~
RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
RaggedToTensor/default_valuew
RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2
RaggedToTensor/Const
#RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorRaggedToTensor/Const:output:0index_lookup/Identity:output:0%RaggedToTensor/default_value:output:0 index_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2%
#RaggedToTensor/RaggedTensorToTensorС
IdentityIdentity,RaggedToTensor/RaggedTensorToTensor:result:06^index_lookup/None_lookup_table_find/LookupTableFindV2*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 2n
5index_lookup/None_lookup_table_find/LookupTableFindV25index_lookup/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: 
щ
Ћ
__inference__traced_save_631
file_prefixR
Nsavev2_index_lookup_index_table_lookup_table_export_values_lookuptableexportv2T
Psavev2_index_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1	
savev2_1_const_1

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_dbe4f74bec3d461c9d8ff7c78490bfe8/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ї
valueBBFlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-values2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
SaveV2/shape_and_slicesЯ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Nsavev2_index_lookup_index_table_lookup_table_export_values_lookuptableexportv2Psavev2_index_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesб
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const_1^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*5
_input_shapes$
": :џџџџџџџџџ:џџџџџџџџџ: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
:џџџџџџџџџ:)%
#
_output_shapes
:џџџџџџџџџ:

_output_shapes
: 
з
Ш
__inference__traced_restore_646
file_prefix
|layer_with_weights_0__index_lookup_layer__table__attributes_table_table_restore_lookuptableimportv2_index_lookup_index_table

identity_1Ђ	RestoreV2ЂRestoreV2_1Ђclayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table_table_restore/LookupTableImportV2
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ї
valueBBFlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-values2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
RestoreV2/shape_and_slicesЕ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes

::*
dtypes
2	2
	RestoreV2у
clayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table_table_restore/LookupTableImportV2LookupTableImportV2|layer_with_weights_0__index_lookup_layer__table__attributes_table_table_restore_lookuptableimportv2_index_lookup_index_tableRestoreV2:tensors:0RestoreV2:tensors:1*	
Tin0*

Tout0	*+
_class!
loc:@index_lookup_index_table*
_output_shapes
 2e
clayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table_table_restore/LookupTableImportV2Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЪ
IdentityIdentityfile_prefix^NoOpd^layer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2

Identityи

Identity_1IdentityIdentity:output:0
^RestoreV2^RestoreV2_1d^layer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :2
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_12Ъ
clayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table_table_restore/LookupTableImportV2clayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:IE
+
_class!
loc:@index_lookup_index_table

_output_shapes
: 
С
У
>__inference_model_layer_call_and_return_conditional_losses_346
input_1
text_vectorization_340
text_vectorization_342	
identity	Ђ*text_vectorization/StatefulPartitionedCallЁ
*text_vectorization/StatefulPartitionedCallStatefulPartitionedCallinput_1text_vectorization_340text_vectorization_342*
Tin
2	*
Tout
2	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_text_vectorization_layer_call_and_return_conditional_losses_3292,
*text_vectorization/StatefulPartitionedCallН
IdentityIdentity3text_vectorization/StatefulPartitionedCall:output:0+^text_vectorization/StatefulPartitionedCall*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 2X
*text_vectorization/StatefulPartitionedCall*text_vectorization/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: 
d
в
__inference__wrapped_model_276
input_1_
[model_text_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_table_handle`
\model_text_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
identity	ЂNmodel/text_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2
$model/text_vectorization/StringLowerStringLowerinput_1*'
_output_shapes
:џџџџџџџџџ2&
$model/text_vectorization/StringLower
+model/text_vectorization/StaticRegexReplaceStaticRegexReplace-model/text_vectorization/StringLower:output:0*'
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2-
+model/text_vectorization/StaticRegexReplaceЪ
 model/text_vectorization/SqueezeSqueeze4model/text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2"
 model/text_vectorization/Squeeze
*model/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2,
*model/text_vectorization/StringSplit/Const
2model/text_vectorization/StringSplit/StringSplitV2StringSplitV2)model/text_vectorization/Squeeze:output:03model/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:24
2model/text_vectorization/StringSplit/StringSplitV2Х
8model/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8model/text_vectorization/StringSplit/strided_slice/stackЩ
:model/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2<
:model/text_vectorization/StringSplit/strided_slice/stack_1Щ
:model/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:model/text_vectorization/StringSplit/strided_slice/stack_2ј
2model/text_vectorization/StringSplit/strided_sliceStridedSlice<model/text_vectorization/StringSplit/StringSplitV2:indices:0Amodel/text_vectorization/StringSplit/strided_slice/stack:output:0Cmodel/text_vectorization/StringSplit/strided_slice/stack_1:output:0Cmodel/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask24
2model/text_vectorization/StringSplit/strided_sliceТ
:model/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:model/text_vectorization/StringSplit/strided_slice_1/stackЦ
<model/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<model/text_vectorization/StringSplit/strided_slice_1/stack_1Ц
<model/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model/text_vectorization/StringSplit/strided_slice_1/stack_2б
4model/text_vectorization/StringSplit/strided_slice_1StridedSlice:model/text_vectorization/StringSplit/StringSplitV2:shape:0Cmodel/text_vectorization/StringSplit/strided_slice_1/stack:output:0Emodel/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Emodel/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask26
4model/text_vectorization/StringSplit/strided_slice_1
?model/text_vectorization/StringSplit/RaggedFromValueRowIds/CastCast;model/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2A
?model/text_vectorization/StringSplit/RaggedFromValueRowIds/Cast§
Amodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1Cast=model/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2C
Amodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1
Imodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ShapeShapeCmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2K
Imodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Shapeр
Imodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2K
Imodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Constх
Hmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/ProdProdRmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Shape:output:0Rmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2J
Hmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Prodр
Mmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater/yё
Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/GreaterGreaterQmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Prod:output:0Vmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2M
Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater
Hmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/CastCastOmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2J
Hmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Castф
Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2M
Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_1е
Gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MaxMaxCmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cast:y:0Tmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2I
Gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Maxи
Imodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2K
Imodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add/yт
Gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/addAddV2Pmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Max:output:0Rmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2I
Gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/addе
Gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/mulMulLmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Cast:y:0Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2I
Gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/mulк
Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MaximumMaximumEmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1:y:0Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2M
Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Maximumо
Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/MinimumMinimumEmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cast_1:y:0Omodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2M
Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Minimumн
Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2M
Kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_2Т
Lmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/BincountBincountCmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cast:y:0Omodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Minimum:z:0Tmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2N
Lmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Bincountв
Fmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum/axisф
Amodel/text_vectorization/StringSplit/RaggedFromValueRowIds/CumsumCumsumSmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/bincount/Bincount:bins:0Omodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2C
Amodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsumт
Jmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2L
Jmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/concat/values_0в
Fmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/concat/axisИ
Amodel/text_vectorization/StringSplit/RaggedFromValueRowIds/concatConcatV2Smodel/text_vectorization/StringSplit/RaggedFromValueRowIds/concat/values_0:output:0Gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/Cumsum:out:0Omodel/text_vectorization/StringSplit/RaggedFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ2C
Amodel/text_vectorization/StringSplit/RaggedFromValueRowIds/concat
Nmodel/text_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2[model_text_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_table_handle;model/text_vectorization/StringSplit/StringSplitV2:values:0\model_text_vectorization_index_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:2P
Nmodel/text_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2
7model/text_vectorization/index_lookup/assert_equal/NoOpNoOp*
_output_shapes
 29
7model/text_vectorization/index_lookup/assert_equal/NoOpш
.model/text_vectorization/index_lookup/IdentityIdentityWmodel/text_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:20
.model/text_vectorization/index_lookup/Identityъ
0model/text_vectorization/index_lookup/Identity_1IdentityJmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ22
0model/text_vectorization/index_lookup/Identity_1А
5model/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 27
5model/text_vectorization/RaggedToTensor/default_valueЉ
-model/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2/
-model/text_vectorization/RaggedToTensor/Const
<model/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor6model/text_vectorization/RaggedToTensor/Const:output:07model/text_vectorization/index_lookup/Identity:output:0>model/text_vectorization/RaggedToTensor/default_value:output:09model/text_vectorization/index_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2>
<model/text_vectorization/RaggedToTensor/RaggedTensorToTensorѓ
IdentityIdentityEmodel/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0O^model/text_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 2 
Nmodel/text_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2Nmodel/text_vectorization/index_lookup/None_lookup_table_find/LookupTableFindV2:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: 
КI
Г
K__inference_text_vectorization_layer_call_and_return_conditional_losses_329

inputsF
Bindex_lookup_none_lookup_table_find_lookuptablefindv2_table_handleG
Cindex_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
identity	Ђ5index_lookup/None_lookup_table_find/LookupTableFindV2Z
StringLowerStringLowerinputs*'
_output_shapes
:џџџџџџџџџ2
StringLowerФ
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*'
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2
StaticRegexReplace
SqueezeSqueezeStaticRegexReplace:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2	
Squeezeg
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
StringSplit/ConstГ
StringSplit/StringSplitV2StringSplitV2Squeeze:output:0StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:2
StringSplit/StringSplitV2
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
StringSplit/strided_slice/stack
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!StringSplit/strided_slice/stack_1
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!StringSplit/strided_slice/stack_2т
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask2
StringSplit/strided_slice
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!StringSplit/strided_slice_1/stack
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#StringSplit/strided_slice_1/stack_1
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#StringSplit/strided_slice_1/stack_2Л
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
StringSplit/strided_slice_1Й
&StringSplit/RaggedFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2(
&StringSplit/RaggedFromValueRowIds/CastВ
(StringSplit/RaggedFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2*
(StringSplit/RaggedFromValueRowIds/Cast_1О
0StringSplit/RaggedFromValueRowIds/bincount/ShapeShape*StringSplit/RaggedFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:22
0StringSplit/RaggedFromValueRowIds/bincount/ShapeЎ
0StringSplit/RaggedFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0StringSplit/RaggedFromValueRowIds/bincount/Const
/StringSplit/RaggedFromValueRowIds/bincount/ProdProd9StringSplit/RaggedFromValueRowIds/bincount/Shape:output:09StringSplit/RaggedFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 21
/StringSplit/RaggedFromValueRowIds/bincount/ProdЎ
4StringSplit/RaggedFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 26
4StringSplit/RaggedFromValueRowIds/bincount/Greater/y
2StringSplit/RaggedFromValueRowIds/bincount/GreaterGreater8StringSplit/RaggedFromValueRowIds/bincount/Prod:output:0=StringSplit/RaggedFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 24
2StringSplit/RaggedFromValueRowIds/bincount/Greaterв
/StringSplit/RaggedFromValueRowIds/bincount/CastCast6StringSplit/RaggedFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 21
/StringSplit/RaggedFromValueRowIds/bincount/CastВ
2StringSplit/RaggedFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2StringSplit/RaggedFromValueRowIds/bincount/Const_1ё
.StringSplit/RaggedFromValueRowIds/bincount/MaxMax*StringSplit/RaggedFromValueRowIds/Cast:y:0;StringSplit/RaggedFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 20
.StringSplit/RaggedFromValueRowIds/bincount/MaxІ
0StringSplit/RaggedFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :22
0StringSplit/RaggedFromValueRowIds/bincount/add/yў
.StringSplit/RaggedFromValueRowIds/bincount/addAddV27StringSplit/RaggedFromValueRowIds/bincount/Max:output:09StringSplit/RaggedFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 20
.StringSplit/RaggedFromValueRowIds/bincount/addё
.StringSplit/RaggedFromValueRowIds/bincount/mulMul3StringSplit/RaggedFromValueRowIds/bincount/Cast:y:02StringSplit/RaggedFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 20
.StringSplit/RaggedFromValueRowIds/bincount/mulі
2StringSplit/RaggedFromValueRowIds/bincount/MaximumMaximum,StringSplit/RaggedFromValueRowIds/Cast_1:y:02StringSplit/RaggedFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 24
2StringSplit/RaggedFromValueRowIds/bincount/Maximumњ
2StringSplit/RaggedFromValueRowIds/bincount/MinimumMinimum,StringSplit/RaggedFromValueRowIds/Cast_1:y:06StringSplit/RaggedFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 24
2StringSplit/RaggedFromValueRowIds/bincount/MinimumЋ
2StringSplit/RaggedFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 24
2StringSplit/RaggedFromValueRowIds/bincount/Const_2Х
3StringSplit/RaggedFromValueRowIds/bincount/BincountBincount*StringSplit/RaggedFromValueRowIds/Cast:y:06StringSplit/RaggedFromValueRowIds/bincount/Minimum:z:0;StringSplit/RaggedFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ25
3StringSplit/RaggedFromValueRowIds/bincount/Bincount 
-StringSplit/RaggedFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-StringSplit/RaggedFromValueRowIds/Cumsum/axis
(StringSplit/RaggedFromValueRowIds/CumsumCumsum:StringSplit/RaggedFromValueRowIds/bincount/Bincount:bins:06StringSplit/RaggedFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2*
(StringSplit/RaggedFromValueRowIds/CumsumА
1StringSplit/RaggedFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 23
1StringSplit/RaggedFromValueRowIds/concat/values_0 
-StringSplit/RaggedFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-StringSplit/RaggedFromValueRowIds/concat/axisЛ
(StringSplit/RaggedFromValueRowIds/concatConcatV2:StringSplit/RaggedFromValueRowIds/concat/values_0:output:0.StringSplit/RaggedFromValueRowIds/Cumsum:out:06StringSplit/RaggedFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ2*
(StringSplit/RaggedFromValueRowIds/concat
5index_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Bindex_lookup_none_lookup_table_find_lookuptablefindv2_table_handle"StringSplit/StringSplitV2:values:0Cindex_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:27
5index_lookup/None_lookup_table_find/LookupTableFindV2^
index_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2 
index_lookup/assert_equal/NoOp
index_lookup/IdentityIdentity>index_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2
index_lookup/Identity
index_lookup/Identity_1Identity1StringSplit/RaggedFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2
index_lookup/Identity_1~
RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
RaggedToTensor/default_valuew
RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2
RaggedToTensor/Const
#RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorRaggedToTensor/Const:output:0index_lookup/Identity:output:0%RaggedToTensor/default_value:output:0 index_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2%
#RaggedToTensor/RaggedTensorToTensorС
IdentityIdentity,RaggedToTensor/RaggedTensorToTensor:result:06^index_lookup/None_lookup_table_find/LookupTableFindV2*
T0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџ:: 2n
5index_lookup/None_lookup_table_find/LookupTableFindV25index_lookup/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: "ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*О
serving_defaultЊ
;
input_10
serving_default_input_1:0џџџџџџџџџO
text_vectorization9
StatefulPartitionedCall:0	џџџџџџџџџџџџџџџџџџtensorflow/serving/predict:P

layer-0
layer_with_weights-0
layer-1
trainable_variables
	variables
regularization_losses
	keras_api

signatures
*#&call_and_return_all_conditional_losses
$_default_save_signature
%__call__"
_tf_keras_modelљ{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TextVectorization", "config": {"name": "text_vectorization", "trainable": true, "dtype": "string", "max_tokens": 5000, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": null, "pad_to_max_tokens": true}, "name": "text_vectorization", "inbound_nodes": [[["input_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["text_vectorization", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TextVectorization", "config": {"name": "text_vectorization", "trainable": true, "dtype": "string", "max_tokens": 5000, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": null, "pad_to_max_tokens": true}, "name": "text_vectorization", "inbound_nodes": [[["input_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["text_vectorization", 0, 0]]}}}
ч"ф
_tf_keras_input_layerФ{"class_name": "InputLayer", "name": "input_1", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "input_1"}}
н
state_variables
	_index_lookup_layer

trainable_variables
	variables
regularization_losses
	keras_api
*&&call_and_return_all_conditional_losses
'__call__" 
_tf_keras_layer{"class_name": "TextVectorization", "name": "text_vectorization", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "config": {"name": "text_vectorization", "trainable": true, "dtype": "string", "max_tokens": 5000, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": null, "pad_to_max_tokens": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
metrics
trainable_variables
	variables
layer_regularization_losses

layers
non_trainable_variables
layer_metrics
regularization_losses
%__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
,
(serving_default"
signature_map
 "
trackable_dict_wrapper
Ѓ
state_variables

_table
trainable_variables
	variables
regularization_losses
	keras_api
*)&call_and_return_all_conditional_losses
*__call__"ѓ
_tf_keras_layerй{"class_name": "IndexLookup", "name": "index_lookup", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "config": {"name": "index_lookup", "trainable": true, "dtype": "string", "max_tokens": 5000, "num_oov_tokens": 1, "vocabulary": null, "reserve_zero": true, "mask_zero": false}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
metrics

trainable_variables
	variables
layer_regularization_losses

layers
non_trainable_variables
layer_metrics
regularization_losses
'__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
@
+_create_resource
,_initialize
-_destroy_resourceR 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
metrics
trainable_variables
	variables
layer_regularization_losses

 layers
!non_trainable_variables
"layer_metrics
regularization_losses
*__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ц2У
>__inference_model_layer_call_and_return_conditional_losses_501
>__inference_model_layer_call_and_return_conditional_losses_452
>__inference_model_layer_call_and_return_conditional_losses_355
>__inference_model_layer_call_and_return_conditional_losses_346Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
м2й
__inference__wrapped_model_276Ж
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *&Ђ#
!
input_1џџџџџџџџџ
к2з
#__inference_model_layer_call_fn_392
#__inference_model_layer_call_fn_519
#__inference_model_layer_call_fn_510
#__inference_model_layer_call_fn_374Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ѕ2ђ
K__inference_text_vectorization_layer_call_and_return_conditional_losses_568Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_text_vectorization_layer_call_fn_577Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0B.
!__inference_signature_wrapper_403input_1
З2ДБ
ЈВЄ
FullArgSpec'
args
jself
jinputs
jinvert
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
З2ДБ
ЈВЄ
FullArgSpec'
args
jself
jinputs
jinvert
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Џ2Ќ
__inference__creator_582
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference__initializer_587
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Б2Ў
__inference__destroyer_592
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
	J
Const4
__inference__creator_582Ђ

Ђ 
Њ " 6
__inference__destroyer_592Ђ

Ђ 
Њ " 8
__inference__initializer_587Ђ

Ђ 
Њ " Ћ
__inference__wrapped_model_276.0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "PЊM
K
text_vectorization52
text_vectorizationџџџџџџџџџџџџџџџџџџ	А
>__inference_model_layer_call_and_return_conditional_losses_346n.8Ђ5
.Ђ+
!
input_1џџџџџџџџџ
p

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ	
 А
>__inference_model_layer_call_and_return_conditional_losses_355n.8Ђ5
.Ђ+
!
input_1џџџџџџџџџ
p 

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ	
 Џ
>__inference_model_layer_call_and_return_conditional_losses_452m.7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ	
 Џ
>__inference_model_layer_call_and_return_conditional_losses_501m.7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ	
 
#__inference_model_layer_call_fn_374a.8Ђ5
.Ђ+
!
input_1џџџџџџџџџ
p

 
Њ "!џџџџџџџџџџџџџџџџџџ	
#__inference_model_layer_call_fn_392a.8Ђ5
.Ђ+
!
input_1џџџџџџџџџ
p 

 
Њ "!џџџџџџџџџџџџџџџџџџ	
#__inference_model_layer_call_fn_510`.7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "!џџџџџџџџџџџџџџџџџџ	
#__inference_model_layer_call_fn_519`.7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "!џџџџџџџџџџџџџџџџџџ	Й
!__inference_signature_wrapper_403.;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"PЊM
K
text_vectorization52
text_vectorizationџџџџџџџџџџџџџџџџџџ	Д
K__inference_text_vectorization_layer_call_and_return_conditional_losses_568e./Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ	
 
0__inference_text_vectorization_layer_call_fn_577X./Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџ	