package javaNT;
import edu.wpi.first.wpilibj.networktables.*;
// import java.util.*;
import java.io.*;

public class NTsend {
    static final String ip = "255.255.255.0";
    static final String ntName = "AutoAim";
    static NetworkTable table;

	public static void main(String[] args) {
		//etable = NetworkTable.getTable(ntName);
		String s = "e";
		String f = "eafegaewrgqeg";
        while (f == "eafegaewrgqeg") { // this ensures it'll only run once while not having to recode the loop later when i get it to work
            System.out.println("startup, reading input\n");

	        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
	        
            try {
		    	while ((s = in.readLine()) != null && s.length() != 0) {
		    		f = s;
		    	    System.out.println(s);
		        }
		    }

            catch (IOException e) {
		    	e.printStackTrace();
            }
		}
    }
}
