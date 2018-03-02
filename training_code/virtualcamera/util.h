/*
  License:
  -----------------------------------------------------------------------------
  Copyright (c) 2017, Gabriel Eilertsen.
  All rights reserved.
  
  Redistribution and use in source and binary forms, with or without 
  modification, are permitted provided that the following conditions are met:
  
  1. Redistributions of source code must retain the above copyright notice, 
     this list of conditions and the following disclaimer.
  
  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
  
  3. Neither the name of the copyright holder nor the names of its contributors
     may be used to endorse or promote products derived from this software 
     without specific prior written permission.
  
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
  POSSIBILITY OF SUCH DAMAGE.
  -----------------------------------------------------------------------------
 
  Description: Utility functions for virtual camera data augmentation.
  Author: Gabriel Eilertsen, gabriel.eilertsen@liu.se
  Date: March 2018
*/


#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <dirent.h>
#include <sys/stat.h>


namespace VCUtil
{

// Recursive search for files in a directory 
void findFiles(const char *name, std::vector<std::string> *fnames, bool verbose = 0, int level = 0)
{
    DIR *dir;
    struct dirent *entry;

    if (!(dir = opendir(name)))
        return;
    if (!(entry = readdir(dir)))
        return;

    do {
        if (entry->d_type == DT_DIR) {
            char path[1024];
            int len = snprintf(path, sizeof(path)-1, "%s/%s", name, entry->d_name);
            path[len] = 0;
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            if (verbose) printf("%*s[%s]\n", level*2, "", entry->d_name);
            findFiles(path, fnames, verbose, level + 1);
        }
        else
        {
            if (verbose) printf("%*s- %s\n", level*2, "", entry->d_name);
            
            char str[1024];
            sprintf(str, "%s/%s", name, entry->d_name);
            fnames->push_back(str);
        }
    } while ((entry = readdir(dir)));
    closedir(dir);
}


// Create a directory
bool createDir(const char *name)
{
    DIR *dir;
    if((dir = opendir(name)))
    {
        closedir(dir);
        return 1;
    }

    int status = mkdir(name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (status == -1)
    {
        printf("Error creating output directory '%s'!\n", name);
        return 0;
    }

    return 1;
}


// Reads 'size' string arguments from the input array
bool readArgument(const int argc, char* args[], std::string name, std::string* arg, int size = 1)
{
    bool found = false;
    
    for (int argIndex = 0; argIndex < argc; ++argIndex)
    {
        std::string tempArg;
        tempArg = args[argIndex];
        if (tempArg == name)
        {
            found = true;
            if (argIndex < argc - size)
            {
                for (int i = 0; i < size; i++)
                {
                    tempArg = args[argIndex + 1];
                    if (tempArg.c_str()[0] == '-')
                    {
                        std::cout << "Warning: Flag '" << name << "' expects " << size << " argument(s)." << std::endl;
                        return false;
                    }
                    arg[i] = tempArg;
                    ++argIndex;
                }
                return true;
            }
            else
            {
                std::cout << "Warning: Flag '" << name << "' expects " << size << " argument(s)." << std::endl;
                return false;
            }
        }
    }
    
    return found;
}

// Reads 'size' float arguments from the input array
bool readArgument(const int argc, char* args[], std::string name, float* arg, int size = 1)
{
    bool found = false;
    
    float* f = new float[size];
    for (int argIndex = 0; argIndex < argc; ++argIndex)
    {
        std::string tempArg;
        tempArg = args[argIndex];
        if (tempArg == name)
        {
            found = true;
            if (argIndex < argc - size)
            {
                for (int i = 0; i < size; i++)
                {
                    tempArg = args[argIndex + 1];
                    sscanf(tempArg.c_str(), "%f", &f[i]);
                    if (tempArg.c_str()[0] == '-' && f[i] >= 0.0f)
                    {
                        std::cout << "Warning: Flag '" << name << "' expects " << size << " argument(s)." << std::endl;
                        delete[] f;
                        return false;
                    }
                    ++argIndex;
                }
            }
            else
            {
                delete[] f;
                std::cout << "Warning: Flag '" << name << "' expects "<< size <<" argument(s)." << std::endl;
                return false;
            }
        }
    }
    
    if (found)
        for (int i = 0; i < size; i++)
            arg[i] = f[i];
    delete[] f;
    
    return found;
}

// Reads 'size' int arguments from the input array
bool readArgument(const int argc, char* args[], std::string name, int* arg, int size = 1)
{
    bool found = false;
    
    for (int argIndex = 0; argIndex < argc; ++argIndex)
    {
        std::string tempArg;
        tempArg = args[argIndex];
        if (tempArg == name)
        {
            found = true;
            if (argIndex < argc - size)
            {                
                int d;
                for (int i = 0; i < size; i++)
                {
                    tempArg = args[argIndex + 1];
                    sscanf(tempArg.c_str(), "%d", &d);
                    arg[i] = d;
                    ++argIndex;
                }
                return true;
            }
            else
            {
                std::cout << "Warning: Flag '" << name << "' expects " << size << " argument(s)." << std::endl;
                return false;
            }
        }
    }
    
    return found;
}

// Reads 'size' long unsigned int arguments from the input array
bool readArgument(const int argc, char* args[], std::string name, long unsigned int* arg, int size = 1)
{
    bool found = false;
    
    for (int argIndex = 0; argIndex < argc; ++argIndex)
    {
        std::string tempArg;
        tempArg = args[argIndex];
        if (tempArg == name)
        {
            found = true;
            if (argIndex < argc - size)
            {                
                int d;
                for (int i = 0; i < size; i++)
                {
                    tempArg = args[argIndex + 1];
                    sscanf(tempArg.c_str(), "%d", &d);
                    arg[i] = d;
                    ++argIndex;
                }
                return true;
            }
            else
            {
                std::cout << "Warning: Flag '" << name << "' expects " << size << " argument(s)." << std::endl;
                return false;
            }
        }
    }
    
    return found;
}

// Reads 'size' bool arguments from the input array
bool readArgument(const int argc, char* args[], std::string name, bool* arg, int size = 1)
{
    int temp[size];
    for (int ind = 0; ind < size; ind++)
        temp[ind] = arg[ind];
    
    bool found = readArgument(argc, args, name, temp, size);
    for (int ind = 0; ind < size; ind++)
        arg[ind] = temp[ind] > 0;
    
    return found;
}

}