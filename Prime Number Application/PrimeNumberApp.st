<?xml version="1.0"?>

<st-source>
<time-stamp>From VisualWorks® Personal Use Edition, 8.3 of July 28, 2017 on February 26, 2020 at 11:02:07 AM</time-stamp>


<class>
<name>PrimeNumberApp</name>
<environment>Smalltalk</environment>
<super>UI.ApplicationModel</super>
<private>false</private>
<indexed-type>none</indexed-type>
<inst-vars>numproc Result1 Result2 ProcessingTime LimitPrimeNumber </inst-vars>
<class-inst-vars></class-inst-vars>
<imports></imports>
<category></category>
<attributes>
<package>PrimeNumber</package>
</attributes>
</class>

<shared-variable>
<name>PT</name>
<environment>PrimeNumberApp</environment>
<private>false</private>
<constant>false</constant>
<category>PT</category>
<attributes>
<package>PrimeNumber</package>
</attributes>
</shared-variable>

<shared-variable>
<name>intercommSobelPtr</name>
<environment>PrimeNumberApp</environment>
<private>false</private>
<constant>false</constant>
<category>intercommSobelPtr</category>
<attributes>
<package>PrimeNumber</package>
</attributes>
</shared-variable>

<!-- -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -->


<methods>
<class-id>PrimeNumberApp class</class-id> <category>interface specs</category>

<body package="PrimeNumber" selector="windowSpec">windowSpec	"Tools.UIPainter new openOnClass: self andSelector: #windowSpec"	&lt;resource: #canvas&gt;	^#(#{UI.FullSpec} 		#window: 		#(#{UI.WindowSpec} 			#properties: #(#{UI.PropertyListDictionary} #sizeType #specifiedSize #positionType #positionSystemDefault #openType #advanced ) 			#label: 'StMPI Image Processing' 			#min: #(#{Core.Point} 20 20 ) 			#max: #(#{Core.Point} 0 0 ) 			#bounds: #(#{Graphics.Rectangle} 710 360 1210 680 ) 			#flags: 4 ) 		#component: 		#(#{UI.SpecCollection} 			#collection: #(				#(#{UI.CompositeSpecCollection} 					#collection: #(						#(#{UI.GroupBoxSpec} 							#layout: #(#{Graphics.LayoutFrame} 0 0 0 0 458 0 72 0 ) 							#name: #MPIControl 							#label: 'MPI Control' ) 						#(#{UI.SpinButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 181 0 29 0 281 0 50 0 ) 							#name: #numproc 							#model: #numproc 							#type: #number 							#low: 2 							#interval: 1 ) 						#(#{UI.LabelSpec} 							#layout: #(#{Graphics.LayoutOrigin} 14 0 30 0 ) 							#name: #numprocLabel 							#label: 'Number of processor to use :' ) ) 					#compositeSpec: 					#(#{UI.CompositeSpec} 						#layout: #(#{Graphics.LayoutFrame} 18 0 9 0 476 0 81 0 ) 						#name: #MPIControlGroup ) ) 				#(#{UI.CompositeSpecCollection} 					#collection: #(						#(#{UI.GroupBoxSpec} 							#layout: #(#{Graphics.LayoutFrame} 0 0 0 0 457 0 71 0 ) 							#name: #Operation 							#label: 'Operation' ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 297 0 28 0 437 0 49 0 ) 							#name: #PrimeNumber 							#model: #PrimeNumber 							#label: 'Calculate Prime Number' 							#defaultable: true ) 						#(#{UI.InputFieldSpec} 							#layout: #(#{Graphics.LayoutFrame} 174 0 28 0 274 0 47 0 ) 							#name: #LimitPrimeNumber 							#model: #LimitPrimeNumber 							#type: #number 							#formatString: '0' ) 						#(#{UI.LabelSpec} 							#layout: #(#{Graphics.LayoutOrigin} 14 0 30 0 ) 							#name: #LimitPrimeNumberLabel 							#label: 							#(#{Kernel.UserMessage} 								#key: nil 								#defaultString: 'Limit of the prime number :' 								#catalogID: #LimitPrimeNumberLabel ) ) ) 					#compositeSpec: 					#(#{UI.CompositeSpec} 						#layout: #(#{Graphics.LayoutFrame} 20 0 90 0 477 0 161 0 ) 						#name: #OperationGroup ) ) 				#(#{UI.CompositeSpecCollection} 					#collection: #(						#(#{UI.GroupBoxSpec} 							#layout: #(#{Graphics.LayoutFrame} 0 0 0 0 459 0 123 0 ) 							#name: #Result 							#label: 'Result' ) 						#(#{UI.LabelSpec} 							#layout: #(#{Graphics.LayoutOrigin} 20 0 28 0 ) 							#name: #ProcessingTimeLabel 							#label: 'Processing Time (s):' ) 						#(#{UI.InputFieldSpec} 							#layout: #(#{Graphics.LayoutFrame} 141 0 26 0 296 0 45 0 ) 							#name: #ProcessingTime 							#model: #ProcessingTime 							#type: #fixedpoint ) 						#(#{UI.InputFieldSpec} 							#layout: #(#{Graphics.LayoutFrame} 142 0 93 0 296 0 112 0 ) 							#name: #Result2 							#model: #Result2 							#type: #number ) 						#(#{UI.LabelSpec} 							#layout: #(#{Graphics.LayoutOrigin} 20 0 93 0 ) 							#name: #Result2Label 							#label: 'Total primes :' ) 						#(#{UI.LabelSpec} 							#layout: #(#{Graphics.LayoutOrigin} 20 0 61 0 ) 							#name: #Result1Label 							#label: 'Largest Primes :' ) 						#(#{UI.InputFieldSpec} 							#layout: #(#{Graphics.LayoutFrame} 142 0 59 0 297 0 78 0 ) 							#name: #Result1 							#model: #Result1 							#type: #number ) ) 					#compositeSpec: 					#(#{UI.CompositeSpec} 						#layout: #(#{Graphics.LayoutFrame} 20 0 173 0 479 0 296 0 ) 						#name: #ResultGroup ) ) ) ) )</body>
</methods>

<!-- -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -->


<methods>
<class-id>PrimeNumberApp</class-id> <category>actions</category>

<body package="PrimeNumber" selector="PrimeNumber">PrimeNumber	| rankPtr ntasksPtr pcsumPtr maxprimePtr start_time end_time limitPtr errcodePtr intercommPrimePtr spawnCount |	rankPtr := CIntegerType int gcCalloc.	StMPI new MPI_Comm_rank: StMPI new MPI_COMM_WORLD with: rankPtr.	ntasksPtr := CIntegerType int gcCalloc.	StMPI new MPI_Comm_size: StMPI new MPI_COMM_WORLD with: ntasksPtr.	pcsumPtr := CIntegerType int gcCalloc.	maxprimePtr := CIntegerType int gcCalloc.	limitPtr := CIntegerType long gcCalloc.	intercommPrimePtr := CIntegerType int gcCalloc.	spawnCount := self numproc value.	errcodePtr := CIntegerType int pointerType gcMalloc: spawnCount.	limitPtr contents: self LimitPrimeNumber value.	spawnCount \\ 2 == 0 &amp; (limitPtr contents \\ spawnCount == 0)		ifTrue: 			[Transcript show: ('&lt;n&gt;Using &lt;1p&gt; process ' expandMacrosWith: spawnCount).			Transcript				show: ('to scan &lt;1p&gt; numbers.' expandMacrosWith: limitPtr contents).			start_time := StMPI new MPI_Wtime.			StMPI new				MPI_Comm_spawn: 'PrimeNumber'				with: StMPI new MPI_ARGV_NULL				with: spawnCount				with: StMPI new MPI_INFO_NULL				with: 0				with: StMPI new MPI_COMM_SELF				with: intercommPrimePtr				with: errcodePtr.			StMPI new				MPI_Bcast: limitPtr				with: 1				with: StMPI new MPI_LONG				with: StMPI new MPI_ROOT				with: intercommPrimePtr contents.			StMPI new				MPI_Recv: maxprimePtr				with: 1				with: StMPI new MPI_INT				with: 0				with: 0				with: intercommPrimePtr contents				with: StMPI new MPI_STATUS_IGNORE.			StMPI new				MPI_Recv: pcsumPtr				with: 1				with: StMPI new MPI_INT				with: 0				with: 0				with: intercommPrimePtr contents				with: StMPI new MPI_STATUS_IGNORE.			end_time := StMPI new MPI_Wtime.			StMPI new MPI_Comm_disconnect: intercommPrimePtr.			self ProcessingTime value: end_time - start_time.			self Result1 value: maxprimePtr contents.			self Result2 value: pcsumPtr contents]		ifFalse: 			[Dialog				warn: ('Sorry - this application requires an even number of process.&lt;n&gt;Evenly divisible into &lt;1p&gt;.  Try 4 or 8 number of process. &lt;n&gt;Click OK to continue.'						expandMacrosWith: self LimitPrimeNumber value)].	^self</body>
</methods>

<methods>
<class-id>PrimeNumberApp</class-id> <category>aspects</category>

<body package="PrimeNumber" selector="LimitPrimeNumber">LimitPrimeNumber	"This method was generated by UIDefiner.  Any edits made here	may be lost whenever methods are automatically defined.  The	initialization provided below may have been preempted by an	initialize method."	^LimitPrimeNumber isNil		ifTrue:			[LimitPrimeNumber := 100 asValue]		ifFalse:			[LimitPrimeNumber]</body>

<body package="PrimeNumber" selector="ProcessingTime">ProcessingTime	"This method was generated by UIDefiner.  Any edits made here	may be lost whenever methods are automatically defined.  The	initialization provided below may have been preempted by an	initialize method."	^ProcessingTime isNil		ifTrue:			[ProcessingTime := 0.00s asValue]		ifFalse:			[ProcessingTime]</body>

<body package="PrimeNumber" selector="Result1">Result1	"This method was generated by UIDefiner.  Any edits made here	may be lost whenever methods are automatically defined.  The	initialization provided below may have been preempted by an	initialize method."	^Result1 isNil		ifTrue:			[Result1 := 0 asValue]		ifFalse:			[Result1]</body>

<body package="PrimeNumber" selector="Result2">Result2	"This method was generated by UIDefiner.  Any edits made here	may be lost whenever methods are automatically defined.  The	initialization provided below may have been preempted by an	initialize method."	^Result2 isNil		ifTrue:			[Result2 := 0 asValue]		ifFalse:			[Result2]</body>

<body package="PrimeNumber" selector="numproc">numproc	"This method was generated by UIDefiner.  Any edits made here	may be lost whenever methods are automatically defined.  The	initialization provided below may have been preempted by an	initialize method."	^numproc isNil		ifTrue:			[numproc := 2 asValue]		ifFalse:			[numproc]</body>
</methods>

<methods>
<class-id>PrimeNumberApp</class-id> <category>initialize-release</category>

<body package="PrimeNumber" selector="initialize">initialize| commandLine argcPtr argvPtrPtr tempPtr argvPtrPtrPtr|	Transcript show: ('&lt;n&gt;&lt;n&gt;&lt;1p&gt;'				expandMacrosWith: 'StMPI Image Processing.').	commandLine := CEnvironment commandLine.	argcPtr := CIntegerType int gcCalloc.	argcPtr contents: commandLine size.	argvPtrPtr := CIntegerType char pointerType gcMalloc: commandLine size.	tempPtr := argvPtrPtr copy.	commandLine do: 			[:arg |			tempPtr contents: arg gcCopyToHeap.			tempPtr += 1].	argvPtrPtrPtr := CIntegerType char pointerType pointerType gcMalloc.	argvPtrPtrPtr contents: argvPtrPtr.	StMPI new MPI_Init: argcPtr with: argvPtrPtrPtr.	argcPtr free.	argvPtrPtr free.	argvPtrPtrPtr free.	^self</body>
</methods>

<methods>
<class-id>PrimeNumberApp</class-id> <category>changing</category>

<body package="PrimeNumber" selector="changeRequest">changeRequestStMPI new MPI_Abort: StMPI new MPI_COMM_WORLD with: 1.	^self</body>
</methods>

</st-source>
