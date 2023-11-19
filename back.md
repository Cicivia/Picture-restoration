```
#include <bits/stdc++.h>

using namespace std;

int main()
{
	int n, x;
	cin >> n >> x;
	int a[31];
	int sum = 0;
	int dp[31][300010] = { 0 };
	for (int i = 1; i <= n; i++)
	{
		cin >> a[i];
		sum += a[i];
	}
	a[0] = 0;
	
	//容量为sum-x的背包
	
	for (int i = 0; i <= n; i++)
	{
		for (int j = 0; j <= sum - x; j++)
		{
			if (j < a[i])
			{
				dp[i][j] = dp[i - 1][j];
			}
			else
			{
				dp[i][j] = max(dp[i - 1][j], a[i] + dp[i - 1][j - a[i]]);
			}
		}
	}
	cout << sum-dp[n][sum - x] << endl;
	return 0;
}
```

