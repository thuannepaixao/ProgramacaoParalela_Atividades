#include <thread>
#include <iostream>
#include <chrono>
#include <mutex>
#include <atomic>


//using namespace std;

//unsigned int contador = 0;

std::atomic < unsigned int > contador (0);

std::mutex locker;


void fun_thread() {
	//locker.lock();
	for (int i = 0; i < 5; i++) {
		printf(" Esperar thread: %d..\n", std::this_thread::get_id());
		
		std::cout << " Esperar thread:" << std::this_thread::get_id() << std::endl;
		
		// Timer Sleep - 500 millise
		std::this_thread::sleep_for(std::chrono::microseconds(500));
		//locker.lock();
		contador++;
		//locker.unlock();
	}
	//locker.unlock();
}

int main() {

	std::thread thread01(fun_thread);
	std::thread thread02(fun_thread);

	thread01.join();
	thread02.join();

	printf("Contador: %d", contador.load());
	
	std::cout << " Contador" << contador << std::endl;
}