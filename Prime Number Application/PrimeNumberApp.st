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

<body package="PrimeNumber" selector="windowSpec">windowSpec
</methods>

<!-- -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -->


<methods>
<class-id>PrimeNumberApp</class-id> <category>actions</category>

<body package="PrimeNumber" selector="PrimeNumber">PrimeNumber
</methods>

<methods>
<class-id>PrimeNumberApp</class-id> <category>aspects</category>

<body package="PrimeNumber" selector="LimitPrimeNumber">LimitPrimeNumber

<body package="PrimeNumber" selector="ProcessingTime">ProcessingTime

<body package="PrimeNumber" selector="Result1">Result1

<body package="PrimeNumber" selector="Result2">Result2

<body package="PrimeNumber" selector="numproc">numproc
</methods>

<methods>
<class-id>PrimeNumberApp</class-id> <category>initialize-release</category>

<body package="PrimeNumber" selector="initialize">initialize
</methods>

<methods>
<class-id>PrimeNumberApp</class-id> <category>changing</category>

<body package="PrimeNumber" selector="changeRequest">changeRequest
</methods>

</st-source>