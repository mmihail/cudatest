//******************************************************************
// Filanem:		MyBignum.h
// Description:	Stripped down big number class (N; *=, +=; >, <)
// Author:		Szakály Tamás (sghctoma@gmail.com)
// Date:		2008.06.16.
//******************************************************************

#pragma once

#include <iostream>
#include <string>
#include <sstream>

typedef unsigned char digit;

class MyBignum
{
private:
	std::string m_sDigits;

	void Reverse(std::string& s)
	{
		std::string ret = "";
		for (int i = (int)s.length() - 1; i >= 0; --i)
			ret += s[i];
		s = ret;
	}

	std::string GetDigit(digit x, digit& o)
	{
		o = x / 10;
		std::stringstream ss;
		ss << x % 10;
		std::string s = ss.str();
		return ss.str();
	}

	std::string Add(std::string a, std::string b)
	{
		std::string temp = "";
		digit overflow = 0;

		size_t i = 0;
		for (; i < a.length() || i < b.length(); ++i)
		{
			if (i < a.length() && i < b.length())
				temp += GetDigit((a[i] - '0') + (b[i] - '0') + overflow, overflow);
			else if (i < a.length())
				temp += GetDigit((a[i] - '0') + overflow, overflow);
			else if (i < b.length())
				temp += GetDigit((b[i] - '0') + overflow, overflow);
		}

		if (overflow)
			temp += '1';

		return temp;
	}

	std::string Multiply(std::string a, std::string b)
	{
		std::string ret;

		for (size_t i = 0; i != a.length(); ++i)
		{
			std::string temp;
			for (size_t k =	0; k < i; ++k) temp += "0";
			digit overflow = 0;

			for (size_t j = 0; j != b.length(); ++j)
				temp += GetDigit((a[i] - '0') * (b[j] - '0') + overflow, overflow);
			
			if (overflow)
			{
				std::stringstream ss;
				ss << (int)overflow;
				temp += ss.str();
			}

			ret = Add(ret, temp);
		}

		return ret;
	}

	bool IsLesser(MyBignum ref)
	{
		int l1 = (int)m_sDigits.length();
		int l2 = (int)ref.m_sDigits.length();

		if (l1 > l2)
			return false;
		else if (l1 < l2)
			return true;

		for (int i = l1 - 1; i >= 0; --i)
			if (m_sDigits[i] > ref.m_sDigits[i])
				return false;
			else if (m_sDigits[i] < ref.m_sDigits[i])
				return true;

		return true;
	}
public:
	MyBignum(std::string s)
	{
		m_sDigits = s;
		Reverse(m_sDigits);
	}

	MyBignum(int i)
	{
		std::stringstream ss;
		ss << i;
		std::string s = ss.str();
		Reverse(s);
		m_sDigits = s;
	}

	std::string ToString()
	{
		std::string ret = m_sDigits;
		Reverse(ret);
		return ret;
	}

	MyBignum operator+= (const MyBignum &y)
	{
		m_sDigits = Add(m_sDigits, y.m_sDigits);
		return *this;
	}

    MyBignum operator*= (const MyBignum &y)
	{
		m_sDigits = Multiply(m_sDigits, y.m_sDigits);
		return *this;
	}

	bool operator< (const MyBignum &x)
	{
		return IsLesser(x);
	}

	bool operator> (const MyBignum &x)
	{
		return !IsLesser(x);
	}
};
