# Before `make install' is performed this script should be runnable with
# `make test'. After `make install' it should work as `perl test.pl'

#########################

# change 'tests => 1' to 'tests => last_test_to_print';

use Test;
BEGIN { plan tests => 1 };

use Algorithm::SVM::DataSet;
use Algorithm::SVM;

ok(1); # If we made it this far, we're ok.

#########################

# Insert your test code below, the Test module is use()ed here so read
# its man page ( perldoc Test ) for help writing this test script.

print("Creating new Algorithm::SVM\n");
my $svm = new Algorithm::SVM(Model => 'sample.model');
ok(ref($svm) ne "", 1);

print("Creating new Algorithm::SVM::DataSet objects\n");
my $ds1 = new Algorithm::SVM::DataSet(Label => 1);
my $ds2 = new Algorithm::SVM::DataSet(Label => 2);
my $ds3 = new Algorithm::SVM::DataSet(Label => 3);
ok(ref($ds1) ne "", 1);
ok(ref($ds2) ne "", 1);
ok(ref($ds3) ne "", 1);

print("Adding attributes to Algorithm::SVM::DataSet objects\n");
my @d1 = (0.0424107142857143, 0.0915178571428571, 0.0401785714285714,
	  0.0156250000000000, 0.0156250000000000, 0.0223214285714286,
	  0.0223214285714286, 0.0825892857142857, 0.1205357142857140,
	  0.0736607142857143, 0.0535714285714286, 0.0535714285714286,
	  0.0178571428571429, 0.0357142857142857, 0.1116071428571430,
	  0.0334821428571429, 0.0223214285714286, 0.0602678571428571,
	  0.0200892857142857, 0.0647321428571429);

my @d2 = (0.0673076923076923, 0.11538461538461500, 0.0480769230769231,
	  0.0480769230769231, 0.00961538461538462, 0.0192307692307692,
	  0.0000000000000000, 0.08653846153846150, 0.1634615384615380,
	  0.0865384615384615, 0.03846153846153850, 0.0288461538461538,
	  0.0192307692307692, 0.01923076923076920, 0.0000000000000000,
	  0.0961538461538462, 0.02884615384615380, 0.0673076923076923,
	  0.0288461538461538, 0.02884615384615380);

my @d3 = (0.0756756756756757, 0.0594594594594595, 0.0378378378378378,
	  0.0216216216216216, 0.0432432432432432, 0.0000000000000000,
	  0.0162162162162162, 0.0648648648648649, 0.1729729729729730,
	  0.0432432432432432, 0.0864864864864865, 0.1297297297297300,
	  0.0108108108108108, 0.0108108108108108, 0.0162162162162162,
	  0.0486486486486487, 0.0324324324324324, 0.0216216216216216,
	  0.0594594594594595, 0.0486486486486487);

$ds1->attribute($_, $d1[$_ - 1]) for(1..scalar(@d1));
$ds2->attribute($_, $d2[$_ - 1]) for(1..scalar(@d2));
$ds3->attribute($_, $d3[$_ - 1]) for(1..scalar(@d3));
ok(1);

print("Checking predictions on loaded model\n");
ok($svm->predict($ds1) == 10,1);
ok($svm->predict($ds2) == 0,1);
ok($svm->predict($ds3) == -10,1);

print("Saving model\n");
ok($svm->save('sample.model.1'), 1);

print("Loading saved model\n");
ok($svm->load('sample.model.1'), 1);

print("Checking NRClass\n");
ok($svm->getNRClass(), 3);

print("Checking model labels\n");
ok($svm->getLabels(), (10, 0, -10));
