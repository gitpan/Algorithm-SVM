use ExtUtils::MakeMaker;

$CC = 'c++';

WriteMakefile('NAME'          => 'Algorithm::SVM',
	      'VERSION_FROM'  => 'lib/Algorithm/SVM.pm',
	      'PREREQ_PM'     => {},
	      ($] >= 5.005 ?
	       (ABSTRACT_FROM => 'lib/Algorithm/SVM.pm',
		AUTHOR        => 'Cory Spencer <cspencer@sfu.ca>') : ()),
	      'OPTIMIZE'      => '-O3',  # segfaults with gcc 2.96 if lower (?)
	      'LIBS'          => '-lm',
	      'CC'            => $CC,
	      'LD'            => '$(CC)',
	      'OBJECT'        => 'SVM.o libsvm.o bindings.o',
	      'XSOPT'         => '-C++');
