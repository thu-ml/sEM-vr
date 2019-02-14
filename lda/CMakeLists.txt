add_library(ldalibs stochastic_count.cpp corpus.cpp utils.cpp
                    flags.cpp base_lda.cpp cvsEM.cpp sgd.cpp gd.cpp
                    goem.cpp bestscvb0.cpp bestscvb1.cpp mfvoem.cpp
                    cvb0oem.cpp cvboem.cpp icvboem.cpp ncvboem.cpp
                    rsgd.cpp smd.cpp svi.cpp)
target_link_libraries(ldalibs gflags)

set(LINK_LIBS ldalibs gflags)

add_executable(test_stochastic_count test_stochastic_count.cpp)
target_link_libraries(test_stochastic_count ${LINK_LIBS})

add_executable(lda main.cpp)
target_link_libraries(lda ${LINK_LIBS})