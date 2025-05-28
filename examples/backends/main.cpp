#include <portableRT/portableRT.h>
#include <array>
#include <iostream>

int main() {

    std::cout << "Printing all compiled backends: ";
    for(auto backend : portableRT::all_backends()){
        std::cout << backend.name << ", ";
    }

    std::cout << std::endl;
    return 0;
}